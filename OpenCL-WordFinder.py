import numpy as np
import pyopencl as cl
import os
import random
from concurrent.futures import ThreadPoolExecutor
import time

def ensure_text_file(file_path, size_mb=100):
    if not os.path.exists(file_path):
        print(f"File not found. Creating {file_path} with {size_mb}MB of random text.")
        num_chars = size_mb * 1024 * 1024
        with open(file_path, 'w', encoding='utf-8') as file:
            chars = (random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(num_chars))
            file.write(''.join(chars))
        print("File created.")
    else:
        print("File already exists.")

def load_text_file(file_path):
    print(f"Loading text file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    text_array = np.array([ord(c) for c in text], dtype=np.uint8)
    print(f"Loaded text file with {len(text_array)} characters.")
    return text_array

def cpu_find_word(text_array, word, start_index, end_index):
    matches = []
    word_length = len(word)
    for idx in range(start_index, min(end_index, len(text_array) - word_length + 1)):
        match = True
        for i in range(word_length):
            if text_array[idx + i] != ord(word[i]):
                match = False
                break
        if match:
            matches.append(idx)
    return matches

def parallel_cpu_find(text_array, word, num_threads):
    chunk_size = len(text_array) // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(cpu_find_word, text_array, word, i * chunk_size, (i + 1) * chunk_size) for i in range(num_threads)]
        results = []
        for future in futures:
            results.extend(future.result())
    return results

def find_word_opencl(text_array, word, context, queue):
    kernel_code = """
    __kernel void find_word(__global const char* data, __global int* results, const int word_length, __constant char* search_word) {
        int idx = get_global_id(0);
        int match = 1;
        for (int i = 0; i < word_length; i++) {
            if (data[idx + i] != search_word[i]) {
                match = 0;
                break;
            }
        }
        results[idx] = match;
    }
    """
    program = cl.Program(context, kernel_code).build()
    kernel = program.find_word

    mf = cl.mem_flags
    data_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=text_array)
    word_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array([ord(c) for c in word], dtype=np.uint8))
    result_array = np.zeros(len(text_array) - len(word) + 1, dtype=np.int32)
    result_buffer = cl.Buffer(context, mf.WRITE_ONLY, result_array.nbytes)

    # Get device optimal settings
    work_group_size = kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, queue.device)
    global_size = (len(text_array) - len(word) + 1 + work_group_size - 1) // work_group_size * work_group_size

    kernel.set_args(data_buffer, result_buffer, np.int32(len(word)), word_buffer)
    cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (work_group_size,))

    cl.enqueue_copy(queue, result_array, result_buffer).wait()

    return global_size, work_group_size, np.where(result_array == 1)[0]

def main(file_path):
    ensure_text_file(file_path)
    text_array = load_text_file(file_path)

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    num_cpu_threads = os.cpu_count() or 8
    print(f"Using {num_cpu_threads} CPU threads.")
    print(f"Using GPU device: {device.name}")

    while True:
        word = input("Enter the word to search for (or 'exit' to quit): ").strip().lower()
        if word == 'exit':
            break

        # Measure GPU Time
        print("Starting GPU search...")
        start_gpu = time.time()
        global_size, work_group_size, gpu_results = find_word_opencl(text_array, word, context, queue)
        end_gpu = time.time()
        print("GPU search completed.")

        # Measure CPU Time
        print("Starting CPU search...")
        start_cpu = time.time()
        cpu_results = parallel_cpu_find(text_array, word, num_cpu_threads)
        end_cpu = time.time()
        print("CPU search completed.")

        print(f"GPU used {global_size // work_group_size} work groups, each with {work_group_size} workers.")
        print(f"GPU found the word at positions: {gpu_results} in {end_gpu - start_gpu:.5f} seconds.")
        print(f"CPU used {num_cpu_threads} threads.")
        print(f"CPU found the word at positions: {cpu_results} in {end_cpu - start_cpu:.5f} seconds.")
    
if __name__ == '__main__':
    file_path = 'random_text.txt'
    main(file_path)

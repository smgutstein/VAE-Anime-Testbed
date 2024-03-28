import configparser
import contextlib
import cProfile
import subprocess as sp
import time


def read_config_file(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config

def is_config_file(filename):
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Check if the line resembles an INI section header
                if line.strip().startswith('[') and line.strip().endswith(']'):
                    return True
                # Check if the line resembles a key-value pair
                if '=' in line:
                    return True
        return False
    except FileNotFoundError:
        return False
    
def get_git_hash():
    hash_str =  sp.check_output(['git', 'log', '-n', '1']).decode("utf-8").strip()
    diff_str = sp.check_output(['git', 'diff']).decode("utf-8").strip()
    output_str = hash_str + '\n\n' + diff_str
    return output_str

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper


def profile_decorator(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        with open(f'profiler_output_{timestamp}.txt', 'w') as f:
            with contextlib.redirect_stdout(f):
                profiler.print_stats(sort='cumtime')

        return result

    return wrapper
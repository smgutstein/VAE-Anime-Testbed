import configparser
import contextlib
import cProfile
import subprocess as sp
import time


def read_config_file(filename):
    '''Reads a configuration file and returns a ConfigParser object.'''
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
    '''Returns the info needed to recreate experiment from git records.'''
    branch_str = sp.check_output(['git', 'branch', '--show-current']).decode("utf-8").strip()
    hash_str =  sp.check_output(['git', 'log', '-n', '1']).decode("utf-8").strip()
    diff_str = sp.check_output(['git', 'diff']).decode("utf-8").strip()
    output_str = "Current Branch: " + branch_str + '\n\n' + hash_str + '\n\n' + diff_str
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

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
    
def delt_add(x, y):
    return x + y    
def delt_sub(x,y):
    return x - y    
def delt_mul(x,y):       
    return x * y
def delt_div(x,y):
    return x / y
    
class delta_generator():
    '''Overly generalized function used to give more flexibility
       in how I would update the kl_adj_factor. The idea is to  
       have a function that can be customized to increase or
       decrease the kl_adj_factor in a variety of ways.'''
    def __init__(self, delta_inc_func, delta_dec_func, delta):

        self.delta = delta
        self.inc_func = self.customize_function(delta_inc_func, 1 + delta)
        self.dec_func = self.customize_function(delta_dec_func, 1 + delta)

    def customize_function(self, base_function, y):
        def custom_function(x):
            return base_function(x, y)
        return custom_function
    
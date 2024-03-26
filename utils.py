import configparser
import subprocess as sp

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
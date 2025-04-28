"""
Custom pyext module for APPS evaluation - simplified implementation of RuntimeModule
"""
import sys
import types
import tempfile
import os
import signal
import subprocess
from typing import Tuple, Any, Dict

class TimeoutException(Exception):
    """Exception raised when code execution times out"""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutException("Code execution timed out")

# Register the signal handler
signal.signal(signal.SIGALRM, timeout_handler)

class RuntimeModule:
    """Simulate the RuntimeModule from original pyext package"""
    
    @staticmethod
    def from_string(module_name: str, path: str, code: str) -> types.ModuleType:
        """
        Create a module from code string.
        
        Args:
            module_name: Name for the module
            path: Path (unused but kept for compatibility)
            code: Python code string
            
        Returns:
            The module with executed code
        """
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
            f.write(code.encode())
            file_path = f.name
        
        try:
            # Register signal handler for timeout
            signal.alarm(5)  # 5 seconds timeout
            
            # Create the module
            module = types.ModuleType(module_name)
            
            # Make the module available in sys.modules
            sys.modules[module_name] = module
            
            # Execute the code in the module's namespace
            with open(file_path, 'r') as f:
                exec(compile(f.read(), file_path, 'exec'), module.__dict__)
            
            # Reset the alarm
            signal.alarm(0)
            
            return module
            
        except Exception as e:
            # Reset the alarm
            signal.alarm(0)
            # Re-raise the exception
            raise e
        
        finally:
            # Clean up the temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

def run_test(code: str, test_input: str = None, timeout: int = 5) -> Tuple[int, str, str]:
    """
    Run Python code with the given input and return the output.
    
    Args:
        code (str): Python code to execute
        test_input (str, optional): Input to provide to the program
        timeout (int, optional): Maximum execution time in seconds
    
    Returns:
        tuple: (return_code, stdout, stderr)
    """
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
        f.write(code.encode())
        file_name = f.name
    
    try:
        # Run the code in a separate process
        process = subprocess.Popen(
            [sys.executable, file_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = process.communicate(input=test_input, timeout=timeout)
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            return 124, "", "Execution timed out"
    
    finally:
        # Clean up the temporary file
        if os.path.exists(file_name):
            os.remove(file_name)
    
    return return_code, stdout, stderr 
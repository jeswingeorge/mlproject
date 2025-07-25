import sys
from src.logger import logging

def error_message_details(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    # Extracting the file name and line number where the error occurred
    filename = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = "Error occurred in script: [{0}] at line number: [{1}] with error message: [{2}]".format(
        filename, line_number, str(error)
    )
    # Returning the formatted error message
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return f"{CustomException.__name__}: {self.error_message}"
    
# if __name__ == "__main__":
#     try:
#         a=1/0  # This will raise a ZeroDivisionError
#     except Exception as e:
#         logging.info("ZeroDivisionError occurred")  
#         raise CustomException(e, sys)

    
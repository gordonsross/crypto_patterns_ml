import logging

def setup_file_logger(file_name):
    logging.basicConfig(
            filename=file_name,
            level=logging.DEBUG,
            format='%(asctime)s: %(levelname)s: %(module)s.%(funcName)s: Line %(lineno)d: %(message)s')
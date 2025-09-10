from src.utils.setup_logging import setup_logger
from src.utils.load_config import get_log_config
import pyarrow.compute as pc
import pyarrow.dataset as ds 
import pyarrow.parquet as pq 
import pandas as pd 
import psutil 
import logging 

#from src.utils.setup_logging import setup_logger

logger = logging.getLogger(__name__)
process_monitor = psutil.Process()
setup_logger(logger, get_log_config())

class DB_Pyarrow:
    def __init__(self, 
                 dir:str,
                 id_col: str = 'id_OpenAlex',
                 format:str = 'parquet',
                 RAM_MAX: int = 4500 * (1024**2)
                 ):
        self.DB = ds.dataset(dir, 
                             format = format)
        self.id_col = id_col
        self.RAM_MAX = RAM_MAX 
        self.previous_mem_use = 0
        self.errors = []
        self.batch_size = 1e6
        self.mem_unit = (1024**2)

    def get_available_mem(self):
        mem = psutil.virtual_memory()
        return min(mem.available, self.RAM_MAX)

    def get_batch_size(self,
                       example_nbytes: int,
                       verbose:int = 0):
        available_mem = self.get_available_mem()
        batch_size = available_mem // example_nbytes
        if batch_size < 1:
            raise MemoryError("Batch size 0, not enough memory")
        if verbose > 0:
            logger.info(f"Available Mem: {available_mem / self.mem_unit:.2f} | Batch size: {batch_size}")
        return batch_size

    def get_mem_use(self, verbose:int = 0):
        current_mem_use = process_monitor.memory_info().rss 
        if verbose > 0:
            logger.info(f"MAllocs: {current_mem_use / self.mem_unit:.2f} (Δ{(current_mem_use - self.previous_mem_use ) / self.mem_unit:.2f})")
        self.previous_mem_use = current_mem_use
        return current_mem_use

    def get_cols_all(self, cols: list):
        scanner = self.DB.scanner(columns = cols, 
                                  batch_size = self.batch_size)
        return scanner

    def get_cols_range(self, 
                       cols:list,
                       pyrange_min: int | float,
                       pyrange_max: int | float,
                       range_col: str,
                       ):
        scanner = self.DB.scanner(filter = (ds.field(range_col) >= pyrange_min) &
                                            (ds.field(range_col) < pyrange_max),
                                  columns = cols,
                                  batch_size = self.batch_size,
                                  )
        return scanner
    
    def get_cols_isin(self,
                       cols: list,
                       isin_array: list,
                       isin_col: str,
                       ):
        scanner = self.DB.scanner(filter = ds.field(isin_col).isin(isin_array),
                                  columns = cols,
                                  batch_size = self.batch_size,
                                  )
        return scanner
    

    
if __name__ == "__main__":
    dir = '~/AbstractTransformer/data/ACS_convert/'
    DB = DB_Pyarrow(dir)


    all_ids = DB.get_cols_all(["id_OpenAlex"])

    referenced_works = DB.get_cols_range(cols = ["id_OpenAlex", "referenced_works_OpenAlex"],
                                         pyrange_max = t_end,
                                         pyrange_min= t_start,
                                         range_col = "publication_date_int")
    




    

        

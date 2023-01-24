import fcsparser
import pandas
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CovidCytofDataset(Dataset):
    """
    Custom dataset implementation for pytorch.
    This implementation aim to load COVID_cyTOF data.
    """
    def __init__(self, metada_file_path: str, fcs_path: str, fcs_samples: int | bool = False) -> None:
        """
        Create a new CovidCytofDataset instance.

        :param metada_file_path: The path to the file that contain the dataset metadata
        :param fcs_path: path to the directory that contain all the .fcs files
        :param fcs_samples: the number of row to randomly import for each .fcs files, use False to load everything
        """
        self.metadata: pd.DataFrame = pd.read_excel(metada_file_path)
        self.fcs: None | pd.DataFrame = None
        self.fcs_samples: int = fcs_samples
        self.fcs_path: str = fcs_path

        self.__load_fcs()
        self.__join_fcs_and_metadata()
        self.__transform_data()

    def __load_fcs(self):
        """
        Load all the .fcs files and merge them into a single pandas DataFrame
        :return:
        """
        print("Loading fcs data:")
        for barcode in tqdm(self.metadata["Kit_Barcode"]):
            data: pd.DataFrame  # only type definition for intellisense
            _, data = fcsparser.parse(self.__get_fcs_filename(barcode), reformat_meta=True)
            data.insert(0, "Kit_Barcode", barcode)

            if self.fcs_samples and len(data) > self.fcs_samples:
                data = data.sample(n=self.fcs_samples)

            self.fcs = pd.concat([self.fcs, data]) if self.fcs is not None else data

    def __get_fcs_filename(self, barcode: str) -> str:
        """
        Return the filename of the .fcs file that correspond to the given barcode
        :param barcode: the barcode of the sample
        :return: the path of the .fcs file
        """
        return f"{self.fcs_path}/{barcode}_Ungated.fcs"

    def __join_fcs_and_metadata(self) -> None:
        """
        Join the metadata and the fcs data into a single pandas DataFrame:
        """
        self.data = pandas.merge(self.metadata, self.fcs, how='inner', on="Kit_Barcode")
        print(self.data.head())

    def __transform_data(self) -> None:
        # @TODO: implement this method
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item):
        # @TODO: implement this method
        pass


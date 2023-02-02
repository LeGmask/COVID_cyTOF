import fcsparser
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy import stats


class CovidCytofDataset(Dataset):
    """
    Customs dataset implementation for pytorch.
    This implementation aims to load COVID_cyTOF data.
    """

    def __init__(self, metada_file_path: str, fcs_path: str, fcs_samples: int | bool = False) -> None:
        """
        Creates a new CovidCytofDataset instance.

        :param metada_file_path: The path to the file containing the dataset metadata
        :param fcs_path: The path to the directory containing all the .fcs files
        :param fcs_samples: The number of row to randomly import for each .fcs files, use False to load everything
        """
        self.metadata: pd.DataFrame = pd.read_excel(metada_file_path)
        self.fcs: None | pd.DataFrame = None
        self.fcs_samples: int | bool = fcs_samples 
        self.fcs_path: str = fcs_path

        self.__load_fcs()
        self.__join_fcs_and_metadata()
        self.__transform_data()

    def __load_fcs(self):
        """
        Loads all the .fcs files and merge them into a single pandas DataFrame
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
        Returns the filename of the .fcs file that corresponds to the given barcode
        :param barcode: the barcode of the sample
        :return: the path of the .fcs file
        """
        return f"{self.fcs_path}/{barcode}_Ungated.fcs"

    def __join_fcs_and_metadata(self) -> None:
        """
        Joins the metadata and the fcs data into a single pandas DataFrame
        """
        self.data = pd.merge(self.metadata, self.fcs, how='inner', on="Kit_Barcode")

    def __transform_data(self) -> None:
        """
        Transforms data : removes useless columns, adds a Label column, normalizes (with Z score) and converts to pytorch format
        """
        self.labels = pd.factorize(self.data["COVID status"])[0]
        self.data.drop(["RecordID", "Kit_Barcode", "COVID status", "Age Group", "Sex", "Time", "Event_length", "Center", "Offset", "Width", "Residual", "beadDist"], axis=1, inplace=True)
        self.data = stats.zscore(self.data)
        self.data = torch.tensor(self.data.to_numpy(), dtype=torch.float32)
        print("done")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item):
        """
        Gets an item (i.e. a cell) by it row number.
        """
        return self.data[item], self.labels[item]
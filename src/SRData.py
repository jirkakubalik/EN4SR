import copy

import numpy as np
import sys
import pandas as pd
import warnings
import glob

warnings.filterwarnings("ignore")
import tensorflow as tf

import SRConfig


class SRData:
    allData = None
    x_data = None
    y_data = None
    x_val = None
    y_val = None
    nVal = None
    x_test = None
    y_test = None
    x_plot = None
    y_plot = None
    miniBatches: list[tuple] = None
    datasetNames: list[str] = None

    forwardpass_data = None
    forwardFullMask = None
    forwardpass_data_boundaries = {}
    forwardpass_masks = {}
    forwardpass_counts = {}

    def generateMinibatches(k: int = 200):
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (SRData.x_data, SRData.y_data)
        )
        train_dataset = train_dataset.shuffle(
            buffer_size=10 * SRData.y_data.shape[0], seed=SRConfig.seed
        ).batch(batch_size=SRConfig.minibatchSize, drop_remainder=True)
        SRData.miniBatches = []
        for i in range(k):
            for _, (tmp_batch_x, tmp_batch_y) in enumerate(train_dataset):
                mini_batch_x = tmp_batch_x
                mini_batch_y = tmp_batch_y
                SRData.miniBatches.append((mini_batch_x, mini_batch_y))
        return

    def read_data_from_file(train_val_file: str = None, test_file=None):
        """
        Reads data from "--train_data" file and splits them into training and validation sets according to the ratio specified by "--train_val" parameter.
        Reads test data from "--test_data" file.
        """
        # -----------------------------------
        # --- Training and validation data
        # -----------------------------------
        if train_val_file is None:
            print(f"No data file defined", file=sys.stderr)
            exit(1)
        # ---
        df = pd.read_csv(train_val_file, header=None, skiprows=0)
        data = df.to_numpy().reshape(df.shape)
        cols = list(range(data.shape[1] - 1))
        # --- Too few data samples
        if data.shape[0] <= 20:
            SRData.x_data = data[:, cols].reshape((data.shape[0], len(cols)))
            SRData.y_data = SRConfig.scaleCoeff * data[:, -1].reshape(
                (data.shape[0], 1)
            )
            SRData.x_val = SRData.x_data
            SRData.y_val = SRData.y_data
        # --- Enough data samples
        else:
            # --- Shuffle training data
            data_ids = np.array(range(data.shape[0]))
            SRConfig.r.shuffle(data_ids)
            data = data[data_ids, :]
            # ---
            # print(f"\nValidation data: {data_ids[:10]} ...")
            # --- Training data
            firstN = int(SRConfig.train_val_split * data.shape[0])
            SRData.x_data = data[:firstN, cols].reshape((firstN, len(cols)))
            SRData.y_data = SRConfig.scaleCoeff * data[:firstN, -1].reshape((firstN, 1))
            # -------------------------------
            # --- Validation data: ORIGINAL
            # -------------------------------
            lastN = data.shape[0] - firstN
            if lastN > 0:
                SRData.x_val = data[data.shape[0] - lastN :, cols].reshape(
                    lastN, len(cols)
                )
                SRData.y_val = SRConfig.scaleCoeff * data[
                    data.shape[0] - lastN :, data.shape[1] - 1
                ].reshape((lastN, 1))
            else:
                SRData.x_val = data[:, cols].reshape(data.shape[0], len(cols))
                SRData.y_val = SRConfig.scaleCoeff * data[:, data.shape[1] - 1].reshape(
                    (data.shape[0], 1)
                )
        SRData.nVal = SRData.x_val.shape[0]
        # ----------------
        # --- Test data
        # ----------------
        if test_file is not None:
            df = pd.read_csv(test_file, header=None, skiprows=0)
            data = df.to_numpy().reshape(df.shape)
            SRData.x_test = data[:, cols].reshape((data.shape[0], len(cols)))
            SRData.y_test = SRConfig.scaleCoeff * data[:, data.shape[1] - 1].reshape(
                (data.shape[0], 1)
            )
        else:
            SRData.x_test = SRData.x_data
            SRData.y_test = SRData.y_data
        # ----------------------------------
        # --- Generate minibatches
        # ----------------------------------
        SRData.generateMinibatches()
        # ---
        print(f"Training+validation data file: {SRConfig.train_data}")
        print(f"\ttraining data: {SRData.x_data.shape}")
        print(f"\tvalidation data: {SRData.x_val.shape}")
        print(f"Test data file: {SRConfig.test_data}")
        print(f"\ttest data: {SRData.x_test.shape}")

    def set_data_for_final_tuning(train_val_file: str):
        """
        Reads data from "--train_data" file and uses all data samples for both the training and validation data.
        """
        # -----------------------------------
        # --- Training and validation data
        # -----------------------------------
        df = pd.read_csv(train_val_file, header=None, skiprows=0)
        data = df.to_numpy().reshape(df.shape)
        cols = list(range(data.shape[1] - 1))
        SRConfig.minibatchSize = data.shape[0]
        # --- Use all data samples
        SRData.x_data = data[:, cols].reshape((data.shape[0], len(cols)))
        SRData.y_data = SRConfig.scaleCoeff * data[:, -1].reshape((data.shape[0], 1))
        SRData.x_val = SRData.x_data
        SRData.y_val = SRData.y_data
        SRData.nVal = SRData.x_val.shape[0]
        # --- Training data
        SRData.forwardpass_data = SRData.x_data[:, :]
        SRData.forwardpass_data_boundaries["train"] = (0, data.shape[0] - 1)
        SRData.forwardpass_counts["train"] = data.shape[0]
        # --- Validation data
        SRData.forwardpass_data_boundaries["valid"] = (
            SRData.forwardpass_data.shape[0],
            SRData.forwardpass_data.shape[0] + SRData.x_val.shape[0] - 1,
        )
        SRData.forwardpass_counts["valid"] = SRData.x_val.shape[0]
        SRData.forwardpass_data = np.append(
            SRData.forwardpass_data, SRData.x_val, axis=0
        )
        # --- Test data
        SRData.forwardpass_data_boundaries["test"] = (
            SRData.forwardpass_data.shape[0],
            SRData.forwardpass_data.shape[0] + SRData.x_test.shape[0] - 1,
        )
        SRData.forwardpass_counts["test"] = SRData.x_test.shape[0]
        SRData.forwardpass_data = np.append(
            SRData.forwardpass_data, SRData.x_test, axis=0
        )
        # ----------------------------------
        # --- Generate minibatches
        # ----------------------------------
        SRData.generateMinibatches(k=3)  # --- k=1 should be ok as well
        # ---
        print(f"Data for final tuning")
        print(f"\ttraining data: {SRData.x_data.shape}")
        print(f"\tvalidation data: {SRData.x_val.shape}")

    @staticmethod
    def init_data(train_val_file: str = None, test_file=None):
        SRData.read_data_from_file(train_val_file=train_val_file, test_file=test_file)
        # --- Training data
        SRData.forwardpass_data = SRData.x_data[: SRConfig.minibatchSize, :]
        SRData.forwardpass_data_boundaries["train"] = (0, SRConfig.minibatchSize - 1)
        SRData.forwardpass_counts["train"] = SRConfig.minibatchSize
        # --- Validation data
        SRData.forwardpass_data_boundaries["valid"] = (
            SRData.forwardpass_data.shape[0],
            SRData.forwardpass_data.shape[0] + SRData.x_val.shape[0] - 1,
        )
        SRData.forwardpass_counts["valid"] = SRData.x_val.shape[0]
        SRData.forwardpass_data = np.append(
            SRData.forwardpass_data, SRData.x_val, axis=0
        )
        # --- Test data
        SRData.forwardpass_data_boundaries["test"] = (
            SRData.forwardpass_data.shape[0],
            SRData.forwardpass_data.shape[0] + SRData.x_test.shape[0] - 1,
        )
        SRData.forwardpass_counts["test"] = SRData.x_test.shape[0]
        SRData.forwardpass_data = np.append(
            SRData.forwardpass_data, SRData.x_test, axis=0
        )
        return SRData

    @staticmethod
    def init_data_continuous(test_file=None):
        # -----------------------------------------------------------------------
        # --- Read data from the first SRConfig.continuousBatch data set files
        # -----------------------------------------------------------------------
        allData: np.array = None
        for i in range(SRConfig.continuousStart, SRConfig.continuousEnd + 1):
            fname = SRData.datasetNames.pop(0)
            df = pd.read_csv(fname, header=None, skiprows=0)
            data = df.to_numpy().reshape(df.shape)
            if allData is None:
                allData = data
            else:
                allData = np.vstack((allData, data))
            # --- keep only unique rows
            allData, _ = np.unique(allData, axis=0, return_index=True)
            if allData.shape[0] >= SRConfig.continuousBatch:
                allData = allData[: SRConfig.continuousBatch, :]
                break
        SRData.allData = copy.deepcopy(allData)
        # -----------------------------------------------------------------------
        # --- Split the data into training and validation set
        # -----------------------------------------------------------------------
        cols = list(range(allData.shape[1] - 1))
        # --- Shuffle training data
        data_ids = np.array(range(allData.shape[0]))
        SRConfig.r.shuffle(data_ids)
        allData = allData[data_ids, :]
        # --- Training data
        firstN = int(SRConfig.train_val_split * allData.shape[0])
        SRData.x_data = allData[:firstN, cols].reshape((firstN, len(cols)))
        SRData.y_data = SRConfig.scaleCoeff * allData[:firstN, -1].reshape((firstN, 1))
        # --- Validation data
        lastN = allData.shape[0] - firstN
        if lastN > 0:
            SRData.x_val = allData[allData.shape[0] - lastN :, cols].reshape(
                lastN, len(cols)
            )
            SRData.y_val = SRConfig.scaleCoeff * allData[
                allData.shape[0] - lastN :, allData.shape[1] - 1
            ].reshape((lastN, 1))
        else:
            SRData.x_val = allData[:, cols].reshape(allData.shape[0], len(cols))
            SRData.y_val = SRConfig.scaleCoeff * allData[
                :, allData.shape[1] - 1
            ].reshape((allData.shape[0], 1))
        SRData.nVal = SRData.x_val.shape[0]
        # -----------------------------------------------------------------------
        # --- Read test data
        # -----------------------------------------------------------------------
        if test_file is not None:
            df = pd.read_csv(test_file, header=None, skiprows=0)
            data = df.to_numpy().reshape(df.shape)
            SRData.x_test = data[:, cols].reshape((data.shape[0], len(cols)))
            SRData.y_test = SRConfig.scaleCoeff * data[:, data.shape[1] - 1].reshape(
                (data.shape[0], 1)
            )
        else:
            SRData.x_test = SRData.x_data
            SRData.y_test = SRData.y_data
        # -----------------------------------------------------------------------
        # --- Set forwardpass variables
        # -----------------------------------------------------------------------
        if SRConfig.minibatchSize < 0:
            SRConfig.minibatchSize = SRData.x_data.shape[0] + SRConfig.minibatchSize
        # --- Training data
        SRData.forwardpass_data = SRData.x_data[: SRConfig.minibatchSize, :]
        SRData.forwardpass_data_boundaries["train"] = (0, SRConfig.minibatchSize - 1)
        SRData.forwardpass_counts["train"] = SRConfig.minibatchSize
        # --- Validation data
        SRData.forwardpass_data_boundaries["valid"] = (
            SRData.forwardpass_data.shape[0],
            SRData.forwardpass_data.shape[0] + SRData.x_val.shape[0] - 1,
        )
        SRData.forwardpass_counts["valid"] = SRData.x_val.shape[0]
        SRData.forwardpass_data = np.append(
            SRData.forwardpass_data, SRData.x_val, axis=0
        )
        # --- Test data
        SRData.forwardpass_data_boundaries["test"] = (
            SRData.forwardpass_data.shape[0],
            SRData.forwardpass_data.shape[0] + SRData.x_test.shape[0] - 1,
        )
        SRData.forwardpass_counts["test"] = SRData.x_test.shape[0]
        SRData.forwardpass_data = np.append(
            SRData.forwardpass_data, SRData.x_test, axis=0
        )
        # ----------------------------------
        # --- Generate minibatches
        # ----------------------------------
        SRData.generateMinibatches()

    @staticmethod
    def reload_data_continuous(new_train_val_file=None, test_file=None):
        # -----------------------------------------------------------------------
        # --- Read and process new data set
        # -----------------------------------------------------------------------
        allData: np.array = SRData.allData
        df = pd.read_csv(new_train_val_file, header=None, skiprows=0)
        data = df.to_numpy().reshape(df.shape)
        allData = np.vstack((allData, data))
        # --- keep only unique rows
        allData, _ = np.unique(allData, axis=0, return_index=True)
        if allData.shape[0] >= SRConfig.continuousBatch:
            allData = allData[
                -SRConfig.continuousBatch :, :
            ]  # --- keep the last SRConfig.continuousBatch data samples
        SRData.allData = copy.deepcopy(allData)
        # -----------------------------------------------------------------------
        # --- Split the data into training and validation set
        # -----------------------------------------------------------------------
        cols = list(range(allData.shape[1] - 1))
        # --- Training data
        firstN = int(SRConfig.train_val_split * allData.shape[0])
        SRData.x_data = allData[:firstN, cols].reshape((firstN, len(cols)))
        SRData.y_data = SRConfig.scaleCoeff * allData[:firstN, -1].reshape((firstN, 1))
        # --- Validation data
        lastN = allData.shape[0] - firstN
        if lastN > 0:
            SRData.x_val = allData[allData.shape[0] - lastN :, cols].reshape(
                lastN, len(cols)
            )
            SRData.y_val = SRConfig.scaleCoeff * allData[
                allData.shape[0] - lastN :, allData.shape[1] - 1
            ].reshape((lastN, 1))
        else:
            SRData.x_val = allData[:, cols].reshape(allData.shape[0], len(cols))
            SRData.y_val = SRConfig.scaleCoeff * allData[
                :, allData.shape[1] - 1
            ].reshape((allData.shape[0], 1))
        SRData.nVal = SRData.x_val.shape[0]
        # -----------------------------------------------------------------------
        # --- Read test data
        # -----------------------------------------------------------------------
        if test_file is not None:
            df = pd.read_csv(test_file, header=None, skiprows=0)
            allData = df.to_numpy().reshape(df.shape)
            SRData.x_test = allData[:, cols].reshape((allData.shape[0], len(cols)))
            SRData.y_test = SRConfig.scaleCoeff * allData[
                :, allData.shape[1] - 1
            ].reshape((allData.shape[0], 1))
        else:
            SRData.x_test = SRData.x_data
            SRData.y_test = SRData.y_data
        # -----------------------------------------------------------------------
        # --- Set forwardpass variables
        # -----------------------------------------------------------------------
        # --- Training data
        SRData.forwardpass_data = SRData.x_data[: SRConfig.minibatchSize, :]
        SRData.forwardpass_data_boundaries["train"] = (0, SRConfig.minibatchSize - 1)
        SRData.forwardpass_counts["train"] = SRConfig.minibatchSize
        # --- Validation data
        SRData.forwardpass_data_boundaries["valid"] = (
            SRData.forwardpass_data.shape[0],
            SRData.forwardpass_data.shape[0] + SRData.x_val.shape[0] - 1,
        )
        SRData.forwardpass_counts["valid"] = SRData.x_val.shape[0]
        SRData.forwardpass_data = np.append(
            SRData.forwardpass_data, SRData.x_val, axis=0
        )
        # --- Test data
        SRData.forwardpass_data_boundaries["test"] = (
            SRData.forwardpass_data.shape[0],
            SRData.forwardpass_data.shape[0] + SRData.x_test.shape[0] - 1,
        )
        SRData.forwardpass_counts["test"] = SRData.x_test.shape[0]
        SRData.forwardpass_data = np.append(
            SRData.forwardpass_data, SRData.x_test, axis=0
        )
        # ----------------------------------
        # --- Generate minibatches
        # ----------------------------------
        SRData.generateMinibatches()

    @staticmethod
    def collect_dataset_sequence(baseName: str = ""):
        SRData.datasetNames = []
        tmp = baseName.split("/")
        fName = tmp[-1]
        filePath = "\\".join(map(str, tmp[:-1]))

        for i in range(
            SRConfig.continuousStart, SRConfig.continuousEnd + 1
        ):  # --- collect all data sets in the given interval
            f = glob.glob(f"{filePath}\\{fName}_{str(i).zfill(3)}.txt")
            if f:
                SRData.datasetNames.extend(f)
        print("Data sets:", SRData.datasetNames)

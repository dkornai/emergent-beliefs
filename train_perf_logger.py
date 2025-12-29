import pandas as pd

class TrainLogger:
    def __init__(self):
        # List of parameter names (strings); initialized after first append
        self.params = None
        
        # Internal storage: list of dicts (each dict maps key -> float)
        self._records = []

    def append(self, record: dict):
        """
        Append a new record (dict of str -> float).
        On the first call, initializes the allowed parameter keys.
        On subsequent calls, checks that the keys match the initialized parameters.
        """
        # --- Validation ---
        if not isinstance(record, dict):
            raise TypeError("Input to append must be a dict.")
        if not all(isinstance(k, str) for k in record.keys()):
            raise TypeError("All keys of the record must be strings.")
        if not all(isinstance(v, (float, int)) for v in record.values()):
            raise TypeError("All values of the record must be floats (or ints).")

        # --- First call: initialize params ---
        if self.params is None:
            self.params = list(record.keys())
        else:
            # Check that keys match exactly
            rec_keys = set(record.keys())
            param_keys = set(self.params)

            if rec_keys != param_keys:
                missing = param_keys - rec_keys
                extra = rec_keys - param_keys
                msg = []
                if missing:
                    msg.append(f"Missing keys: {missing}")
                if extra:
                    msg.append(f"Unexpected keys: {extra}")
                raise KeyError("Record keys do not match initialized params. " + "; ".join(msg))

        # Store record
        self._records.append(record.copy())

    def to_dataframe(self):
        """
        Convert the stored list of dicts into a pandas DataFrame.
        Each row corresponds to one append call.
        """
        return pd.DataFrame(self._records)

    def save_csv(self, path: str):
        """
        Save the stored data as a CSV file using pandas DataFrame conversion.
        """
        df = self.to_dataframe()
        df.index = df.index + 1
        df.index.name = "index"
        df.to_csv(path, index=True)

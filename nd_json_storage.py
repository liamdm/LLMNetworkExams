from typing import Dict, List, Any

import pandas

class NDJsonSaver:
    """
    NDJSON store that:
    - loads the file once at init,
    - keeps a DataFrame in memory,
    - stores keys under the "__key" column,
    - prevents duplicate rows based on key.
    """

    KEY_COLUMN = "__key"

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.df = self._load_initial_df()

    # ---------------- Internal helpers ----------------

    def _load_initial_df(self) -> pandas.DataFrame:
        """Load the NDJSON file once, or create an empty DF."""
        if not os.path.exists(self.output_path):
            return pandas.DataFrame(columns=[self.KEY_COLUMN])

        try:
            df = pandas.read_json(self.output_path, lines=True)
            # Ensure __key column exists
            if self.KEY_COLUMN not in df.columns:
                raise ValueError(f"NDJSON file missing '{self.KEY_COLUMN}' column.")
            return df
        except Exception:
            # Fail safely with a clean empty DF
            return pandas.DataFrame(columns=[self.KEY_COLUMN])

    def _save_df(self):
        """Persist the DataFrame to NDJSON."""
        self.df.to_json(self.output_path, orient="records", lines=True)

    def _key_equals(self, row_key: Dict[str, str]) -> pandas.Series:
        """Return a mask of rows whose __key dict matches row_key exactly."""
        if self.df.empty:
            return pandas.Series([False])

        # Compare dicts directly (safe & simple)
        return self.df[self.KEY_COLUMN].apply(lambda k: k == row_key)

    # ---------------- Public API ----------------

    def rows(self) -> List[Dict[str, Any]]:
        """Return all records (each a dict)."""
        return self.df.to_dict(orient="records")

    def row_exists(self, row_key: Dict[str, str]) -> bool:
        """Check if a row with this key exists in memory."""
        return self._key_equals(row_key).any()

    def add_row(self, row_key: Dict[str, str], row_data: Dict[str, Any]):
        """
        Add a row if it doesn't exist.
        Row structure:
            {
                "__key": { ... },
                ... row_data ...
            }
        """
        if self.row_exists(row_key):
            return  # Already exists, do nothing

        row = {self.KEY_COLUMN: row_key, **row_data}

        # Append to in-memory DF
        self.df = pandas.concat([self.df, pandas.DataFrame([row])], ignore_index=True)

        # Persist updated DF
        self._save_df()

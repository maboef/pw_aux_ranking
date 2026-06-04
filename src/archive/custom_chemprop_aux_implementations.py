# Some of the customized solutions to allow for Chemprop to use the aux_mask besides the standard 

class _MolGraphDatasetMixin:
    def __len__(self) -> int:
        return len(self.data)

    @cached_property
    def _Y(self) -> np.ndarray:
        """the raw targets of the dataset"""
        return np.array([d.y for d in self.data], float)

    @property
    def Y(self) -> np.ndarray:
        """the (scaled) targets of the dataset"""
        return self.__Y

    @Y.setter
    def Y(self, Y: ArrayLike):
        self._validate_attribute(Y, "targets")

        self.__Y = np.array(Y, float)

    @cached_property
    def _X_d(self) -> np.ndarray:
        """the raw extra descriptors of the dataset"""
        return np.array([d.x_d for d in self.data])

    @property
    def X_d(self) -> np.ndarray:
        """the (scaled) extra descriptors of the dataset"""
        return self.__X_d

    @X_d.setter
    def X_d(self, X_d: ArrayLike):
        self._validate_attribute(X_d, "extra descriptors")

        self.__X_d = np.array(X_d)

    @property
    def weights(self) -> np.ndarray:
        return np.array([d.weight for d in self.data])

    @property
    def gt_mask(self) -> np.ndarray:
        return np.array([d.gt_mask for d in self.data])

    @property
    def lt_mask(self) -> np.ndarray:
        return np.array([d.lt_mask for d in self.data])

    @property
    def aux_mask(self) -> np.ndarray:
        return np.array([d.aux_mask for d in self.data])

    @property
    def t(self) -> int | None:
        return self.data[0].t if len(self.data) > 0 else None

    @property
    def d_xd(self) -> int:
        """the extra molecule descriptor dimension, if any"""
        return 0 if self.X_d[0] is None else self.X_d.shape[1]

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.data]

    def normalize_targets(self, scaler: StandardScaler | None = None) -> StandardScaler:
        """Normalizes the targets of this dataset using a :obj:`StandardScaler`

        The :obj:`StandardScaler` subtracts the mean and divides by the standard deviation for
        each task independently. NOTE: This should only be used for regression datasets.

        Returns
        -------
        StandardScaler
            a scaler fit to the targets.
        """

        if scaler is None:
            scaler = StandardScaler().fit(self._Y)

        self.Y = scaler.transform(self._Y)

        return scaler

    def normalize_inputs(
        self, key: str = "X_d", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        VALID_KEYS = {"X_d"}
        if key not in VALID_KEYS:
            raise ValueError(f"Invalid feature key! got: {key}. expected one of: {VALID_KEYS}")

        X = self.X_d if self.X_d[0] is not None else None

        if X is None:
            return scaler

        if scaler is None:
            scaler = StandardScaler().fit(X)

        self.X_d = scaler.transform(X)

        return scaler

    def reset(self):
        """Reset the atom and bond features; atom and extra descriptors; and targets of each
        datapoint to their initial, unnormalized values."""
        self.__Y = self._Y
        self.__X_d = self._X_d

    def _validate_attribute(self, X: np.ndarray, label: str):
        if not len(self.data) == len(X):
            raise ValueError(
                f"number of molecules ({len(self.data)}) and {label} ({len(X)}) "
                "must have same length!"
            )

import torch


class DataTransform:
    def __init__(self, is_train, use_volume=False, additional_features=[]):
        self.is_train = is_train
        self.use_volume = use_volume
        # Feature keys for the model input tensor (no Timestamp)
        self.feature_keys = ['Open', 'High', 'Low', 'Close']
        if use_volume:
            self.feature_keys.append('Volume')
        self.feature_keys += additional_features
        # All keys including Timestamp (for output tracking, not features)
        self.keys = ['Timestamp'] + self.feature_keys
        print(f"Feature keys (model input): {self.feature_keys}")


    def __call__(self, window):
        data_list = []
        output = {}

        # Reference price for per-window normalization: first Close in window
        close_values = torch.tensor(window.get('Close').tolist())
        ref_price = close_values[0]
        if ref_price == 0:
            ref_price = torch.tensor(1.0)

        # Compute volume mean for normalization (exclude target row to match inference)
        if self.use_volume:
            vol_values = torch.tensor(window.get('Volume').tolist())
            vol_mean = vol_values[:-1].mean()  # only feature rows, not target
            if vol_mean == 0:
                vol_mean = torch.tensor(1.0)

        if 'Timestamp_orig' in window.keys() and 'Timestamp_orig' not in self.keys:
            self.keys.append('Timestamp_orig')

        for key in self.keys:
            data = torch.tensor(window.get(key).tolist())

            # Store raw values for target computation before normalization
            output[key] = data[-1]
            output[f'{key}_old'] = data[-2]

            # Skip Timestamp — not a model feature
            if key == 'Timestamp' or key == 'Timestamp_orig':
                continue

            # Per-window normalization
            if key == 'Volume':
                data = data / vol_mean
            elif key in ('Open', 'High', 'Low', 'Close'):
                data = data / ref_price

            data_list.append(data[:-1].reshape(1, -1))

        features = torch.cat(data_list, 0)
        output['features'] = features
        return output

    def set_initial_seed(self, seed):
        self.rng.seed(seed)

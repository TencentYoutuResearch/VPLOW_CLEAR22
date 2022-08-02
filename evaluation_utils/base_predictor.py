"""
PLEASE REMEMBER TO UPDATE THE FILENAME AND PATH IN `aicrowd.yaml` AS NEEDED.
"""


class BaseCLEARPredictor:
    def prediction_setup(self, models_path):
        """
        You can do any preprocessing required for your codebase here :
            like loading your models into memory, etc.
        """

        raise NotImplementedError

    def prediction(self, image_file_path: str):
        """
        This function will be called for all the flight during the evaluation.
        NOTE: In case you want to load your model, please do so in `prediction_setup` function.
        """

        raise NotImplementedError

import keras.callbacks as callbacks
import tierkenner_ressources.visualizer as visualizer
from tierkenner_ressources import Config

class HistoryVisualizer(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}
        self.tmp_config = Config.Config()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        visualizer.save_training_visualization(
            history=self.history,
            dst_accuracy=self.tmp_config.plot_accuracy_current_filepath(),
            dst_loss=self.tmp_config.plot_loss_current_filepath()
        )
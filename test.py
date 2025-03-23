from keras.api.models import load_model
from keras.api.utils import plot_model

model = load_model("models/opinion_detection_baseline_model.keras")

plot_model(model, to_file="plots/baseline_opinion_detection/model_architecture.png", show_shapes=True, show_layer_names=True, show_layer_activations=True)
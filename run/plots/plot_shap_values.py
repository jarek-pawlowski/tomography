# %%
import shap
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.datasets import MeasurementDataset
from src.model import Regressor

# %%
batch_size = 2000
test_dataset = MeasurementDataset(root_path='./data/val/')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model_path = './models/regressor_input_drop_20.pt'
model_params = {
    'input_dim': 16,
    'output_dim': 1,
    'layers': 2,
    'hidden_size': 128,
    'input_dropout': 0.2
}
model = Regressor(**model_params)
model.load(model_path)

tensors, labels = next(iter(test_loader))
ent_tensors = tensors[labels.squeeze() > 0.5]

shap_explainer = shap.DeepExplainer(model, ent_tensors)

shap_values = shap_explainer.shap_values(tensors)

shap_values.shape

shape_values = shap_values[..., 0]

shap.summary_plot(shape_values, show=False)
plt.savefig('entangled_shap_summary_plot.png')




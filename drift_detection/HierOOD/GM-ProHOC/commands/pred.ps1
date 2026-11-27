. "$PSScriptRoot\set_vars.ps1"
python "$REPOROOT\fit_node_densities.py" --config "$CONFIG"
python "$REPOROOT\hierarchical_density_inference.py" --config "$CONFIG"
python "$REPOROOT\export_result_csv.py" --input "$REPOROOT\outputs\fgvc-aircraft-weihims-resnet50\hinference_density.result"

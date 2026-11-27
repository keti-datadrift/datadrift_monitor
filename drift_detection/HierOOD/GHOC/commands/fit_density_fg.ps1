. "$PSScriptRoot\feature_gen_set_vars.ps1"
python "$REPOROOT\fit_node_densities.py" --config "$CONFIG" --feature-gen-config "$FEATURE_GEN_CONFIG"

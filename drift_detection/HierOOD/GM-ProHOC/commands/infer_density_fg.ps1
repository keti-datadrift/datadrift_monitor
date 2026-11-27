. "$PSScriptRoot\feature_gen_set_vars.ps1"
python "$REPOROOT\hierarchical_density_inference.py" --config "$CONFIG" --feature-gen-config "$FEATURE_GEN_CONFIG"

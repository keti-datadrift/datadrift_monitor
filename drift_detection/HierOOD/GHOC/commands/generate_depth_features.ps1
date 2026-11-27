. "$PSScriptRoot\feature_gen_set_vars.ps1"
python "$REPOROOT\generate_depth_features.py" --config "$CONFIG" --feature-gen-config "$FEATURE_GEN_CONFIG"

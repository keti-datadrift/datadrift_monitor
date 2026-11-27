import argparse
import importlib
import json
import os
from collections import defaultdict

import torch

from libs.hierarchy import Hierarchy
from libs import hierarchy_metrics as hm
from libs.utils.hierarchy_utils import (expected_hdist,
                                        gen_flat2node_map,
                                        get_avg_hdist,
                                        get_children_and_group_maps,
                                        get_hdist_matrix,
                                        get_multidepth_classes,
                                        get_path_indices)

parser = argparse.ArgumentParser()

parser.add_argument("--hierarchy", type=str, required=True)
parser.add_argument("--basedir", type=str, required=True)
parser.add_argument("--id_split", type=str, required=True)
parser.add_argument("--uncertainty_methods", nargs="+", required=True)
parser.add_argument("--betas", nargs="+", default=[0])
parser.add_argument("--farood", nargs="+", default=[])
parser.add_argument("--depth_alpha", nargs="+", type=float, default=None)
parser.add_argument("--depth_beta", nargs="+", type=float, default=None)
parser.add_argument("--beta_rule",
                    choices=["ones", "inv", "inv_log", "inv_sqrt"],
                    default="ones")
parser.add_argument("--node_alpha_json", type=str, default=None)
parser.add_argument("--node_beta_json", type=str, default=None)
parser.add_argument("--result_suffix", type=str, default="")
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--beta_schedule",
                    choices=["constant", "inverse_depth", "exp_decay", "linear_decay"],
                    default="constant")
parser.add_argument("--schedule_beta0", type=float, default=1.0)
parser.add_argument("--beta_gamma", type=float, default=0.5)
parser.add_argument("--beta_k", type=float, default=0.5)
parser.add_argument("--beta_min", type=float, default=0.0)
parser.add_argument("--temperature_schedule",
                    choices=["constant", "linear_increase", "exp_increase", "linear_decrease", "exp_decrease", "depth_vector"],
                    default="constant")
parser.add_argument("--temperature_t0", type=float, default=1.0)
parser.add_argument("--temperature_k", type=float, default=0.5)
parser.add_argument("--temperature_r", type=float, default=1.5)
parser.add_argument("--temperature_vector_json", type=str, default=None)


def get_id_classes(id_classes_fn):
    with open(id_classes_fn, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return sorted(line for line in lines if line)


def _resolve_depth_vector(values, expected_len, default=1.0):
    if values is None:
        return [default] * expected_len
    if len(values) == 1:
        return [float(values[0])] * expected_len
    if len(values) != expected_len:
        raise ValueError(
            f"Expected {expected_len} depth weights, got {len(values)}"
        )
    return [float(x) for x in values]


def _load_weight_map(json_path):
    if json_path is None:
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): float(v) for k, v in data.items()}


def _resolve_node_weight_vectors(weight_map, parent_names_by_depth, device):
    if weight_map is None:
        return None

    vectors = []
    for parent_names in parent_names_by_depth:
        vectors.append(
            torch.tensor([weight_map.get(name, 1.0) for name in parent_names],
                         device=device,
                         dtype=torch.float32)
        )
    return vectors


def build_uncertainty_args(args, evaluator, method):
    score_depths = evaluator.score_depths
    depth_alpha = _resolve_depth_vector(args.depth_alpha, score_depths)
    depth_beta = _resolve_depth_vector(args.depth_beta, score_depths)

    u_args = {
        "depth_alpha": depth_alpha,
        "depth_beta": depth_beta,
        "beta_rule": args.beta_rule,
        "beta_schedule": args.beta_schedule,
        "beta0": args.schedule_beta0,
        "beta_gamma": args.beta_gamma,
        "beta_k": args.beta_k,
        "beta_min": args.beta_min,
    }

    if method == "node_weighted_norm":
        node_alpha_map = _load_weight_map(args.node_alpha_json)
        node_beta_map = _load_weight_map(args.node_beta_json)

        u_args["node_alpha_by_depth"] = _resolve_node_weight_vectors(
            node_alpha_map, evaluator.parent_names_by_depth, evaluator.device
        )
        u_args["node_beta_by_depth"] = _resolve_node_weight_vectors(
            node_beta_map, evaluator.parent_names_by_depth, evaluator.device
        )

    return u_args


def resolve_temperature(depth,
                        temperature_schedule="constant",
                        temperature_t0=1.0,
                        temperature_k=0.5,
                        temperature_r=1.5,
                        temperature_vector=None):
    d = depth + 1
    if temperature_schedule == "depth_vector":
        if temperature_vector is None:
            raise ValueError("temperature_vector is required when temperature_schedule='depth_vector'")
        if depth >= len(temperature_vector):
            raise ValueError(
                f"temperature_vector length mismatch: requested depth index {depth}, "
                f"but only {len(temperature_vector)} values were provided"
            )
        return max(1e-6, float(temperature_vector[depth]))
    if temperature_schedule == "constant":
        return float(temperature_t0)
    if temperature_schedule == "linear_increase":
        return float(temperature_t0) + float(temperature_k) * (d - 1)
    if temperature_schedule == "exp_increase":
        return float(temperature_t0) * (float(temperature_r) ** (d - 1))
    if temperature_schedule == "linear_decrease":
        return max(1e-6, float(temperature_t0) - float(temperature_k) * (d - 1))
    if temperature_schedule == "exp_decrease":
        return max(1e-6, float(temperature_t0) / (float(temperature_r) ** (d - 1)))
    raise ValueError(f"Unknown temperature_schedule: {temperature_schedule}")


def build_softmax_with_temperature(logits,
                                   temperature_schedule="constant",
                                   temperature_t0=1.0,
                                   temperature_k=0.5,
                                   temperature_r=1.5,
                                   temperature_vector=None):
    max_height = len(logits)
    softmax = []
    for height_index, logit in enumerate(logits):
        # H0 participates in the deepest local score, H1 in the shallower one.
        depth = max_height - height_index - 2
        if depth >= 0:
            temperature = resolve_temperature(depth,
                                              temperature_schedule=temperature_schedule,
                                              temperature_t0=temperature_t0,
                                              temperature_k=temperature_k,
                                              temperature_r=temperature_r,
                                              temperature_vector=temperature_vector)
        else:
            temperature = 1.0
        softmax.append(torch.softmax(logit / temperature, dim=-1))
    return softmax


def fuse_predictions(softmax,
                     hierarchy,
                     multi_classes,
                     children_maps,
                     group_sizes,
                     path_indices,
                     flat2node_map,
                     uncertainty_method,
                     uncertainty_args):

    max_height = len(softmax)
    n_samples = softmax[0].size(0)
    device = softmax[0].device

    comp_sums = []

    for depth in range(max_height - 1):
        height = max_height - depth - 1
        n_parents = len(multi_classes[depth])

        single_element_mask = group_sizes[depth] == 1
        children_map = children_maps[depth]
        p = softmax[height - 1]

        local_args = dict(uncertainty_args)
        local_args.update({
            "depth": depth,
            "num_classes": p.size(1),
            "parent_names": multi_classes[depth],
        })

        result, p_comp = uncertainty_method(p,
                                            children_map,
                                            group_sizes[depth],
                                            n_samples,
                                            n_parents,
                                            device=device,
                                            **local_args)

        mapped_single_mask = single_element_mask[children_map]

        p_comp[:, single_element_mask] = 0.0
        result[:, mapped_single_mask] = 1.0

        comp_sums.append(p_comp)
        softmax[height - 1].copy_(result)

    expanded_probs = [p[:, path_indices[i]] for i, p in enumerate(reversed(softmax))]
    stacked_probs = torch.stack(expanded_probs, dim=-1)
    cumulative_probs = torch.cumprod(stacked_probs, dim=-1)

    results = []

    for height in range(max_height):
        n_classes = len(multi_classes[-height - 1])
        depth = max_height - height - 1

        intermediate_prod = torch.zeros(n_samples, n_classes, device=device)
        intermediate_prod[:, path_indices[depth]] = cumulative_probs[:, :, depth]

        if height > 0:
            intermediate_prod = intermediate_prod * comp_sums[depth]

        results.append(intermediate_prod)

    results = torch.cat(results, dim=1)

    results_merged = torch.zeros(n_samples, len(hierarchy.id_node_list), device=device)
    results_merged.scatter_add_(1, flat2node_map.expand(n_samples, -1), results)

    psums = torch.sum(results_merged, dim=-1)
    assert torch.allclose(psums, torch.ones_like(psums), atol=1e-4)

    return results_merged


def get_root_probability(softmax_probs, beta):
    top_probs = softmax_probs[-1]
    entropy_probs = softmax_probs[0]

    eps = 1e-9
    entropy = -torch.sum(entropy_probs * torch.log(entropy_probs + eps), dim=1)

    p_ = torch.cat([top_probs, beta * entropy.unsqueeze(-1)], dim=1)
    new_sums = torch.sum(p_, dim=-1, keepdim=True)
    top_probs_rescaled = p_ / new_sums

    return top_probs_rescaled[:, -1]


def predict_classes(val_logits,
                    ood_logits,
                    heights_val,
                    heights_ood,
                    id_hierarchy,
                    multi_classes):

    ood_preds = []
    for i, h in enumerate(heights_ood):
        _, pred = ood_logits[h][i, :].max(dim=0)
        c = multi_classes[-(h + 1)][int(pred)]
        full_class = id_hierarchy.id_node_list.index(c)
        ood_preds.append(full_class)

    val_preds = []
    for i, h in enumerate(heights_val):
        _, pred = val_logits[h][i, :].max(dim=0)
        c = multi_classes[-(h + 1)][int(pred)]
        full_class = id_hierarchy.id_node_list.index(c)
        val_preds.append(full_class)

    return torch.tensor(val_preds), torch.tensor(ood_preds)


def get_results(preds,
                node_labels,
                id_hierarchy,
                dists_mats=None):

    hmet = hm.HierarchicalPredAccuracy(id_hierarchy, track_hdist=True)
    hmet.update_state(preds.long(), node_labels, dists_mats=dists_mats)

    hd = hmet.result_hierarchy_distances()
    return {
        "acc": hmet.result(),
        "balanced_acc": hmet.result_balanced_accuracy(),
        "hdist": hd,
        "avg_hdist": get_avg_hdist(hd),
        "balanced_hdist": hmet.result_balanced_hierarchy_distance(),
        "class_hdists": hmet.result_class_hdists(),
    }


class HInferenceEvaluator:

    def __init__(self, args):
        id_classes = get_id_classes(args.id_split)
        id_hierarchy = Hierarchy(id_classes, args.hierarchy)
        ood_classes = id_hierarchy.ood_train_classes

        self.max_depth = id_hierarchy._max_depth
        self.score_depths = self.max_depth - 1
        print("Depth: ", self.max_depth)
        print("Gather scores...")

        data = defaultdict(list)
        logits = defaultdict(list)
        softmax = defaultdict(list)
        nr_samples = {}

        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but no CUDA device is available")
        self.device = args.device
        dsets = ["val", "ood"] + args.farood

        for dset in dsets:
            for h in range(self.max_depth):
                height_dir = os.path.join(args.basedir, f"H{h}")
                fname = f"{dset}-preds.out"
                data[dset].append(torch.load(os.path.join(height_dir, fname)))
                logits[dset].append(data[dset][h]["logits"].to(self.device))
                softmax[dset].append(torch.softmax(data[dset][h]["logits"], dim=-1).to(self.device))
            nr_samples[dset] = logits[dset][0].size(0)

        node_labels = {}
        for dset in dsets:
            if dset == "val":
                val2node_map = id_hierarchy.gen_ds2node_map(id_classes)
                node_labels[dset] = val2node_map[data["val"][0]["targets"]]
            elif dset == "ood":
                ood2node_map = id_hierarchy.gen_ds2node_map(ood_classes)
                node_labels[dset] = ood2node_map[data["ood"][0]["targets"]]
            else:
                root_idx = id_hierarchy.id_node_list.index("root")
                node_labels[dset] = torch.full((nr_samples[dset],), root_idx, dtype=torch.long)

        multi_classes = get_multidepth_classes(id_hierarchy, id_classes)

        flattened_classes = [item for sublist in reversed(multi_classes) for item in sublist]
        self.flattened_classes = flattened_classes
        self.flat2node = gen_flat2node_map(id_hierarchy, self.flattened_classes, self.device)

        children_maps, group_sizes = get_children_and_group_maps(id_hierarchy,
                                                                 multi_classes,
                                                                 device=self.device)
        self.children_maps = children_maps
        self.group_sizes = group_sizes

        leaf_height = 0
        path_indices = get_path_indices(id_hierarchy, multi_classes, leaf_height)
        self.path_indices = [torch.tensor(x, dtype=torch.long, device=self.device) for x in path_indices]

        gt_dists_mat, pred_dists_mat = get_hdist_matrix(id_hierarchy,
                                                        range(len(id_hierarchy.id_node_list)),
                                                        return_pair=True)
        self.hdist_mat = (gt_dists_mat + pred_dists_mat).to(self.device)

        self.hierarchy = id_hierarchy
        self.gt_dists_mat = gt_dists_mat.long()
        self.pred_dists_mat = pred_dists_mat.long()
        self.multi_classes = multi_classes
        self.dsets = dsets
        self.data = data
        self.logits = logits
        self.softmax = softmax
        self.nr_samples = nr_samples
        self.node_labels = node_labels
        self.parent_names_by_depth = [multi_classes[depth] for depth in range(self.score_depths)]
        self.score_num_classes_by_depth = [len(multi_classes[depth + 1]) for depth in range(self.score_depths)]

    def multi_predict(self,
                      logits,
                      softmax=None,
                      u_method=None,
                      u_args=None,
                      min_hdist=False,
                      beta=1.0,
                      temperature_args=None):

        if u_args is None:
            u_args = {}
        if temperature_args is None:
            temperature_args = {}

        logits = [x.detach().clone() for x in logits]
        if temperature_args:
            softmax = build_softmax_with_temperature(logits, **temperature_args)
        else:
            softmax = [x.detach().clone() for x in softmax]

        fused_p = fuse_predictions(softmax,
                                   self.hierarchy,
                                   self.multi_classes,
                                   self.children_maps,
                                   self.group_sizes,
                                   self.path_indices,
                                   self.flat2node,
                                   u_method,
                                   u_args)

        root_probs = get_root_probability(softmax, beta)
        root_idx = self.hierarchy.id_node_list.index("root")

        assert torch.allclose(fused_p[:, root_idx], torch.zeros_like(fused_p[:, root_idx]))

        fused_p = fused_p * (1 - root_probs.view(-1, 1))
        fused_p[:, root_idx] = root_probs

        psums = torch.sum(fused_p, dim=-1)
        assert torch.allclose(psums, torch.ones_like(psums), atol=1e-4)

        if min_hdist:
            neg_dists = -1.0 * expected_hdist(fused_p, self.hdist_mat)
            _, preds = neg_dists.max(dim=-1)
        else:
            _, preds = fused_p.max(dim=-1)

        return preds

    def predict_and_eval(self, **kwargs):
        res = {}
        for dset in self.dsets:
            preds = self.multi_predict(self.logits[dset], self.softmax[dset], **kwargs)
            res[dset] = get_results(preds.to("cpu"),
                                    self.node_labels[dset],
                                    self.hierarchy,
                                    dists_mats=(self.gt_dists_mat, self.pred_dists_mat))
        return res

    def predict_oracle(self):
        ood_depths = torch.empty(self.nr_samples["ood"], dtype=torch.int)

        for i, ood_label in enumerate(self.node_labels["ood"]):
            class_name = self.hierarchy.id_node_list[ood_label]
            ancestors = self.hierarchy.node_ancestors[class_name]
            ood_depths[i] = len(ancestors)

        ood_heights_oracle = self.max_depth - ood_depths
        val_heights_oracle = torch.zeros(self.nr_samples["val"], dtype=torch.long)

        print("Evaluating oracle...")
        val_preds, ood_preds = predict_classes(self.logits["val"],
                                               self.logits["ood"],
                                               val_heights_oracle,
                                               ood_heights_oracle,
                                               self.hierarchy,
                                               self.multi_classes)

        res_val = get_results(val_preds,
                              self.node_labels["val"],
                              self.hierarchy,
                              dists_mats=(self.gt_dists_mat, self.pred_dists_mat))
        res_ood = get_results(ood_preds,
                              self.node_labels["ood"],
                              self.hierarchy,
                              dists_mats=(self.gt_dists_mat, self.pred_dists_mat))
        return {"val": res_val, "ood": res_ood}

    def predict_leafmodel(self):
        print("Evaluating leaf model...")

        val_heights_leafs = torch.zeros(self.nr_samples["val"], dtype=torch.long)
        ood_heights_leafs = torch.zeros(self.nr_samples["ood"], dtype=torch.long)

        val_preds, ood_preds = predict_classes(self.logits["val"],
                                               self.logits["ood"],
                                               val_heights_leafs,
                                               ood_heights_leafs,
                                               self.hierarchy,
                                               self.multi_classes)

        res_val = get_results(val_preds,
                              self.node_labels["val"],
                              self.hierarchy,
                              dists_mats=(self.gt_dists_mat, self.pred_dists_mat))
        res_ood = get_results(ood_preds,
                              self.node_labels["ood"],
                              self.hierarchy,
                              dists_mats=(self.gt_dists_mat, self.pred_dists_mat))
        return {"val": res_val, "ood": res_ood}


def build_run_metadata(args, evaluator):
    temperature_vector = None
    if getattr(args, "temperature_vector_json", None):
        with open(args.temperature_vector_json, "r", encoding="utf-8") as f:
            temperature_vector = [float(x) for x in json.load(f)]
    return {
        "uncertainty_methods": list(args.uncertainty_methods),
        "betas": [float(x) for x in args.betas],
        "depth_alpha": _resolve_depth_vector(args.depth_alpha, evaluator.score_depths),
        "depth_beta": _resolve_depth_vector(args.depth_beta, evaluator.score_depths),
        "beta_rule": args.beta_rule,
        "node_alpha_json": args.node_alpha_json,
        "node_beta_json": args.node_beta_json,
        "device": args.device,
        "beta_schedule": args.beta_schedule,
        "schedule_beta0": args.schedule_beta0,
        "beta_gamma": args.beta_gamma,
        "beta_k": args.beta_k,
        "beta_min": args.beta_min,
        "temperature_schedule": args.temperature_schedule,
        "temperature_t0": args.temperature_t0,
        "temperature_k": args.temperature_k,
        "temperature_r": args.temperature_r,
        "temperature_vector": temperature_vector,
        "score_depths": evaluator.score_depths,
        "score_num_classes_by_depth": evaluator.score_num_classes_by_depth,
        "parent_names_by_depth": evaluator.parent_names_by_depth,
    }


def main(args):
    hinf = HInferenceEvaluator(args)

    allres = {
        "metadata": build_run_metadata(args, hinf),
        "leafmodel": hinf.predict_leafmodel(),
        "oracle": hinf.predict_oracle(),
    }

    betas = [float(x) for x in args.betas]
    score_module = importlib.import_module("libs.utils.score_util")

    for method in args.uncertainty_methods:
        print(f"Evaluating {method}")
        method_fun = getattr(score_module, method)
        method_args = build_uncertainty_args(args, hinf, method)
        temperature_args = {
            "temperature_schedule": args.temperature_schedule,
            "temperature_t0": args.temperature_t0,
            "temperature_k": args.temperature_k,
            "temperature_r": args.temperature_r,
            "temperature_vector": None,
        }
        if args.temperature_vector_json is not None:
            with open(args.temperature_vector_json, "r", encoding="utf-8") as f:
                temperature_args["temperature_vector"] = [float(x) for x in json.load(f)]

        for beta in betas:
            res = hinf.predict_and_eval(u_method=method_fun,
                                        u_args=method_args,
                                        temperature_args=temperature_args,
                                        min_hdist=False,
                                        beta=beta)
            allres[f"{method}_beta{beta}"] = res

            res = hinf.predict_and_eval(u_method=method_fun,
                                        u_args=method_args,
                                        temperature_args=temperature_args,
                                        min_hdist=True,
                                        beta=beta)
            allres[f"{method}_minhdist_beta{beta}"] = res

    hierarchy_name = os.path.splitext(os.path.basename(args.hierarchy))[0]
    suffix = f"-{args.result_suffix}" if args.result_suffix else ""
    if args.output_path:
        out_path = os.path.abspath(args.output_path)
        out_dir = os.path.dirname(out_path)
    else:
        out_dir = os.path.abspath(os.path.join(os.getcwd(), "results"))
        out_path = os.path.join(out_dir, f"hinference-{hierarchy_name}{suffix}.result")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(allres, out_path)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main(parser.parse_args())

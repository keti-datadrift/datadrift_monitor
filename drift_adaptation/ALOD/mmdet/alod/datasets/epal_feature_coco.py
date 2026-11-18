import mmcv
import numpy as np

from methods.epal.features import FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class EPALFeatureCocoDataset(CocoDataset):
    def _det2json(self, results):
        """Convert EPAL compact feature results to image-level json."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            if not isinstance(result, dict):
                raise TypeError(
                    'EPALFeatureCocoDataset expects dict results, got %r' %
                    (type(result), ))

            bboxes = np.asarray(result['det_bboxes'])
            labels = np.asarray(result['det_labels'])
            features = result['epal_features']
            miss_features = np.asarray(result['epal_miss_features'])

            if bboxes.ndim != 2 or bboxes.shape[1] != 5 or labels.shape != (len(bboxes),):
                raise ValueError('Invalid EPAL detection shapes')
            if miss_features.shape != (2,):
                raise ValueError('Invalid EPAL miss feature shape')
            for key in ('E_cls', 'E_loc'):
                if np.asarray(features[key]).shape != (len(bboxes),):
                    raise ValueError('Invalid EPAL feature shape: %s' % key)
            arrays = [bboxes, labels, miss_features, features['E_cls'], features['E_loc']]
            if any(not np.isfinite(array).all() for array in arrays):
                raise ValueError('EPAL export contains non-finite values')

            final_detections = []
            for i in range(bboxes.shape[0]):
                data = dict()
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['category_id'] = self.cat_ids[int(labels[i])]
                data['score'] = float(bboxes[i][4])
                data['E_cls'] = float(features['E_cls'][i])
                data['E_loc'] = float(features['E_loc'][i])
                final_detections.append(data)

            json_results.append(
                dict(
                    image_id=img_id,
                    final_detections=final_detections,
                    miss_features=dict(
                        R_amt=float(miss_features[0]),
                        R_conf=float(miss_features[1]))))
        return json_results

    def results2json(self, results, outfile_prefix):
        if len(results) != len(self):
            raise ValueError('EPAL results must cover every dataset image')
        result_files = dict()
        if not results or isinstance(results[0], dict):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.json'
            mmcv.dump(dict(schema=FEATURE_SCHEMA, schema_version=FEATURE_SCHEMA_VERSION,
                           images=json_results), result_files['bbox'])
        else:
            raise TypeError('invalid type of EPAL feature results')
        return result_files

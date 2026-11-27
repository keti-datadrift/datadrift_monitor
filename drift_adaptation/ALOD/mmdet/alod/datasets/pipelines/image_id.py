from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class AddImageIdToMeta:
    """Expose the dataset image id as a collected image meta key."""

    def __call__(self, results):
        img_info = results.get('img_info', {})
        if 'id' not in img_info:
            raise KeyError('AddImageIdToMeta requires img_info["id"]')
        results['image_id'] = img_info['id']
        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'

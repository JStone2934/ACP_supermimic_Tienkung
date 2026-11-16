import unittest
import numpy as np

from tools.prepare_stage1_dataset import select_labels


class Stage1DatasetTests(unittest.TestCase):
    def labels(self):
        return dict(contact_info=np.arange(24).reshape(3, 4, 2, 1),
                    quality_pass=np.array([True, False, True]), source_variants=np.array([2, 4, 9]),
                    body_pos_w=np.arange(24).reshape(4, 2, 3),
                    contact_pos_w=np.arange(72).reshape(3, 4, 2, 3),
                    motion_hash=np.array('old-motion'), terrain_hash=np.array('old-terrain'))

    def test_variant_label_alignment_and_shared_frame_data(self):
        original = self.labels()
        selected = select_labels(original, 2, 'motion', 'terrain')
        np.testing.assert_equal(selected['contact_info'], original['contact_info'][2:3])
        np.testing.assert_equal(selected['contact_pos_w'], original['contact_pos_w'][2:3])
        np.testing.assert_equal(selected['body_pos_w'], original['body_pos_w'])
        self.assertEqual(selected['source_variants'].tolist(), [9])
        self.assertEqual(selected['terrain_hash'].item(), 'terrain')
        self.assertEqual(original['terrain_hash'].item(), 'old-terrain')

    def test_refuses_rejected_and_out_of_range_slots(self):
        for slot in (-1, 1, 3):
            with self.assertRaises(ValueError):
                select_labels(self.labels(), slot, 'm', 't')

    def test_refuses_misaligned_variant_array(self):
        labels = self.labels()
        labels['contact_pos_w'] = labels['contact_pos_w'][:2]
        with self.assertRaises(ValueError):
            select_labels(labels, 2, 'm', 't')


if __name__ == '__main__':
    unittest.main()

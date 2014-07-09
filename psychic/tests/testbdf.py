import unittest
import os.path
import numpy as np
import psychic
from psychic.dataformats.bdf import BDFReader, BDFWriter, le_to_int24, int24_to_le, load_bdf, save_bdf
from ..markers import markers_to_events, biosemi_find_ghost_markers

class TestConversion(unittest.TestCase):
  def setUp(self):
    self.know_values = [
      ((0, 0, 0), 0),
      ((1, 0, 0), 1),
      ((255, 255, 127), (1 << 23) - 1),
      ((0, 0, 128), -(1 << 23)),
      ((255, 255, 255), -1)]
    self.ints = [0, 1, (1 << 23) - 1, -(1 << 23), -1]

  def test_list_conversion(self):
    bytes = list(reduce(lambda x, y: x + y, # make flat list of bytes
      [bs for (bs, i) in self.know_values]))
    ints = [i for (bs, i) in self.know_values]

    self.assertEqual(list(le_to_int24(bytes)), ints)
    self.assertEqual(list(int24_to_le(ints)), bytes)

class TestBDFReader(unittest.TestCase):
  def setUp(self):
    self.bdf = BDFReader(open(psychic.find_data_path('sine-256Hz.bdf'), 'rb'))

  def test_read_all(self):
    b = self.bdf
    eeg = b.read_all()

    # check size
    self.assertEqual(eeg.shape[1], 
      b.bdf.header['n_records'] * max(b.bdf.header['n_samples_per_record']))
    self.assertEqual(eeg.shape[0], 17)

    # check frequency peak at 2.6Hz (not 3Hz as mentioned on biosemi.nl!)
    eeg_fft = 2 * np.abs(np.fft.rfft(eeg[:-1, :], axis=1)) / eeg.shape[1]
    freqs = np.fft.fftfreq(eeg.shape[1], 1./256)
    peak_freq = freqs[eeg_fft[:,1:].argmax(axis=1)]
    np.testing.assert_almost_equal(peak_freq, np.ones(16) * 2.61, 2)

class TestBDFWriter(unittest.TestCase):
  def setUp(self):
    self.d = load_bdf(psychic.find_data_path('sine-256Hz.bdf'))
    r = BDFReader(open(psychic.find_data_path('sine-256Hz.bdf'), 'rb'))
    self.header = r.bdf.header
    r.close()

    w = BDFWriter(psychic.find_data_path('sine-256Hz-test.bdf'),
          header=self.header)
    w.write(self.d)
    w.close()

  def test_header(self):
    with open(psychic.find_data_path('sine-256Hz.bdf'), "rb") as f:
      file1 = f.read()

    with open(psychic.find_data_path('sine-256Hz-test.bdf'), "rb") as f:
      file2 = f.read()

    header_length = 4608
    self.assertEqual(file1[:header_length], file2[:header_length])

  def test_data(self):
    d2 = load_bdf(psychic.find_data_path('sine-256Hz-test.bdf'))
    self.assertTrue(np.allclose(self.d.data, d2.data, atol=0.05))

class TestBDF(unittest.TestCase):
  def test_load(self):
    d = load_bdf(psychic.find_data_path('sine-256Hz.bdf'))

    # test labels
    targets = [['A%d' % (i + 1) for i in range(16)]]
    self.assertEqual(d.feat_lab, targets)
    self.assertEqual(d.cl_lab, ['class0', 'class1'])

    # test ids ~ time
    self.assertAlmostEqual(d.ids[0, 256 + 1], 1, 2)

    # test dims
    self.assertEqual(d.nfeatures, 16)
    self.assertEqual(d.ninstances, 60 * 256)
    
    self.assertEqual(d.extra, {})

  def test_save(self):
    d = load_bdf(os.path.join('data', 'sine-256Hz.bdf'))
    save_bdf(d, os.path.join('data', 'sine-256Hz-test.bdf'))
    d2 = load_bdf(os.path.join('data', 'sine-256Hz-test.bdf'))

    self.assertEqual(d.feat_lab, d2.feat_lab)
    self.assertTrue(np.allclose(d.data, d2.data, atol=0.0001))
    self.assertTrue(np.allclose(d.labels, d2.labels))
    self.assertTrue(np.allclose(d.ids, d2.ids))
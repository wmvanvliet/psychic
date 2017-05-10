# -*- coding: utf-8 -*-
import unittest
from ..dataformats.edf import *
from psychic import find_data_path

class TestEDFBaseReader(unittest.TestCase):
  def test_synthetic_content(self):
    '''
    Test EDF reader using artifical EDF dataset. Note that this is not an
    EDF+ dataset and as such does not contain annotations. Annotations decoding
    is separately tested, *but not from a real file*!.
    '''
    reader = BaseEDFReader(
      open(find_data_path('sine3Hz_block0.2Hz.edf'), 'rb'))
    reader.read_header()

    h = reader.header
    # check important header fields
    self.assertEqual(h['label'], ['3Hz +5/-5 V', '0.2Hz Blk 1/0uV'])
    self.assertEqual(h['units'], ['V', 'uV'])
    self.assertEqual(h['contiguous'], True)

    fs = np.asarray(h['n_samples_per_record']) / h['record_length']

    # get records
    recs = list(zip(*reader.records()))
    time = recs[0]
    signals = recs[1]
    annotations = recs[2]

    # check EDF+ fields that are *not present in this file*
    np.testing.assert_equal(time, np.zeros(11) * np.nan)
    self.assertEqual(annotations, tuple([[]] * 11))

    # check 3 Hz sine wave
    sine, block = [np.hstack(s) for s in zip(*signals)]
    target = 5 * np.sin(3 * np.pi * 2 * np.arange(0, sine.size) / fs[0])
    assert np.max((sine - target) ** 2) < 1e-4

    # check .2 Hz block wave
    target = np.sin(.2 * np.pi * 2 * np.arange(1, block.size + 1) / fs[1]) >= 0
    assert np.max((block - target) ** 2) < 1e-4

  def test_tal(self):
    mult_annotations = '+180\x14Lights off\x14Close door\x14\x00'
    with_duration = '+1800.2\x1525.5\x14Apnea\x14\x00'
    test_unicode = '+180\x14€\x14\x00\x00'

    # test annotation with duration
    self.assertEqual(tal(with_duration), [(1800.2, 25.5, ['Apnea'])])

    # test multiple annotations
    self.assertEqual(tal('\x00' * 4 + with_duration * 3), 
      [(1800.2, 25.5, ['Apnea'])] * 3)

    # test multiple annotations for one time point
    self.assertEqual(tal(mult_annotations), 
      [(180., 0., ['Lights off', 'Close door'])])

    # test unicode support
    self.assertEqual(tal(test_unicode), [(180., 0., ['€'])])

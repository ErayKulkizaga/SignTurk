import unittest

from text_processing import GrammarCorrector, SignTextPipeline


class GrammarPipelineTests(unittest.TestCase):
    def test_rule_based_sentence_assembly(self):
        sentence = GrammarCorrector().correct(["ben", "okul", "gitmek"])
        self.assertEqual(sentence, "Ben okula gidiyorum.")

    def test_pipeline_can_disable_tts_per_request(self):
        result = SignTextPipeline().correct(
            ["merhaba", "ben", "su", "istemek"],
            synthesize_audio=False,
        )
        self.assertEqual(result.sentence, "Merhaba, ben su istiyorum.")
        self.assertEqual(result.tts_status, "disabled")
        self.assertIsNone(result.audio_path)


if __name__ == "__main__":
    unittest.main()

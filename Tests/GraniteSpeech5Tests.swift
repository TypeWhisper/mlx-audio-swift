import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing

@testable import MLXAudioSTT

// The Granite Speech 5.0 2B tests from upstream's GraniteSpeechFamilyTests (#303):
// RoPE config decoding, YaRN RoPE and the GPT2 tokenizer pre-tokenizer fix.
@Suite(.serialized)
struct GraniteSpeech5Tests {
    @Test func ropeSchemasAndRoundTrip() throws {
        for key in ["rope_scaling", "rope_parameters"] {
            let json =
                "{\"\(key)\":{\"rope_type\":\"yarn\",\"factor\":4,\"original_max_position_embeddings\":4096,\"rope_theta\":20000}}"
            let config = try JSONDecoder().decode(
                GraniteSpeechTextConfig.self, from: Data(json.utf8))
            #expect(config.ropeScaling?.ropeType == "yarn")
            #expect(config.ropeScaling?.factor == 4)
            #expect(config.ropeTheta == 20000)
            let decoded = try JSONDecoder().decode(
                GraniteSpeechTextConfig.self, from: JSONEncoder().encode(config))
            #expect(decoded.ropeScaling?.originalMaxPositionEmbeddings == 4096)
        }
        let flat = try JSONDecoder().decode(
            GraniteSpeechTextConfig.self,
            from: Data(#"{"rope_type":"yarn","factor":4}"#.utf8))
        #expect(flat.ropeScaling?.factor == 4)
    }

    @Test func gpt2TokenizerUsesItsDeclaredPreTokenizer() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let data =
            #"{"model":{"type":"BPE","vocab":{"<unk>":0,"?":1,"Ċ":2,"?Ċ":3},"merges":["? Ċ"],"unk_token":"<unk>"},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"\\S+\\s*"},"behavior":"Removed","invert":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}}"#
        try Data(data.utf8).write(to: directory.appendingPathComponent("tokenizer.json"))
        for (name, expected) in [("GPT2Tokenizer", [1, 2]), ("TokenizersBackend", [3])] {
            let config = ["tokenizer_class": name, "unk_token": "<unk>"]
            try JSONSerialization.data(withJSONObject: config).write(
                to: directory.appendingPathComponent("tokenizer_config.json"))
            let tokenizer = try await loadGraniteSpeechTokenizer(from: directory)
            #expect(tokenizer.encode(text: "?\n") == expected)
        }
    }

    @Test func yarnRotatesAtAbsoluteCachePositions() throws {
        let config = try JSONDecoder().decode(
            GraniteSpeechTextConfig.self,
            from: Data(#"{"rope_scaling":{"rope_type":"yarn","factor":4}}"#.utf8))
        let rope = GraniteSpeechRotaryEmbedding(dimensions: 128, config: config)
        #expect(abs(rope.magnitude - (1 + 0.1 * log(Float(4)))) < 1e-7)
        let x = sin(arange(3 * 128).asType(.float32)).reshaped(1, 1, 3, 128)
        let whole = rope(x, offset: 5000)
        let tail = rope(x[0..., 0..., 2..., 0...], offset: 5002)
        #expect(abs(whole[0..., 0..., 2..., 0...] - tail).max().item(Float.self) < 1e-5)
        let frequencies = try #require(rope.frequencies)
        #expect(abs(frequencies[0].item(Float.self) - 1) < 1e-7)
        #expect(
            abs(frequencies[-1].item(Float.self) / pow(Float(10000), Float(126) / 128) - 4) < 1e-5)
    }
}

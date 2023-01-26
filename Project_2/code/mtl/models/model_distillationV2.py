import torch
import torch.nn.functional as F

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p, SelfAttention, DecoderOld


class Model_Distillation_V2(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc
        ch_out = sum(outputs_desc.values())

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=True,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.aspp_semseg = ASPP(ch_out_encoder_bottleneck, 256)

        self.decoder_semseg = DecoderDeeplabV3p(256, ch_out_encoder_4x, list(outputs_desc.values())[0])

        self.aspp_depth = ASPP(ch_out_encoder_bottleneck, 256)

        self.decoder_depth = DecoderDeeplabV3p(256, ch_out_encoder_4x, list(outputs_desc.values())[1])

        self.self_attention_semseg = SelfAttention(256, 256)

        self.self_attention_depth = SelfAttention(256, 256)

        self.last_decoder_semseg = DecoderDeeplabV3p(256, 256, list(outputs_desc.values())[0])

        self.last_decoder_depth = DecoderDeeplabV3p(256, 256, list(outputs_desc.values())[1])

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        # print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        features_tasks_semseg = self.aspp_semseg(features_lowest)

        predictions_4x_semseg, features_4x_semseg = self.decoder_semseg(features_tasks_semseg, features[4])

        predictions_1x_semseg = F.interpolate(predictions_4x_semseg, size=input_resolution, mode='bilinear', align_corners=False)

        features_tasks_depth = self.aspp_depth(features_lowest)

        predictions_4x_depth, features_4x_depth = self.decoder_depth(features_tasks_depth , features[4])

        predictions_1x_depth  = F.interpolate(predictions_4x_depth , size=input_resolution, mode='bilinear', align_corners=False)


        features_sa_semseg = self.self_attention_semseg(features_tasks_depth)

        predictions_4x_final_semseg, _ = self.last_decoder_semseg(features_sa_semseg, features_4x_semseg)

        predictions_final_semseg = F.interpolate(predictions_4x_final_semseg, size=input_resolution, mode='bilinear', align_corners=False)

        features_sa_depth = self.self_attention_depth(features_tasks_semseg)

        predictions_4x_final_depth, _ = self.last_decoder_depth(features_sa_depth, features_4x_depth)

        predictions_final_depth = F.interpolate(predictions_4x_final_depth, size=input_resolution, mode='bilinear', align_corners=False)

        out = {}
        offset = 0

        out[list(self.outputs_desc.keys())[0]] = [predictions_1x_semseg, predictions_final_semseg]
        out[list(self.outputs_desc.keys())[1]] = [predictions_1x_depth, predictions_final_depth]
        #for task, num_ch in self.outputs_desc.items():
          #  out[task] = predictions_1x[:, offset:offset+num_ch, :, :]
          #  offset += num_ch

        return out


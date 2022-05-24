import torch
import torchvision.models
import torch.nn as nn


class Backbone_MultiView(torch.nn.Module):
    """
        Backbone Network: DenseNet121
    """
    def __init__(self, backbone_name, img_size=224, in_chans=3, embed_dim=768, feature_size=None):
        """
        Args:
        :param backbone_name (string): backbone network structure, default="densenet121"
        :param img_size (int, tuple): input image size, default:224
        :param in_chans (int): number of input channels, default:3
        :param embed_dim (int): embedding dimension, default:768
        :param feature_size:
        """
        super().__init__()

        model = getattr(torchvision.models, backbone_name)(pretrained=True)
        self.model = torch.nn.Sequential(*list(model.features.children()))[:-1]
        # Freeze model
        for param in model.parameters():
            param.requires_grad = False

        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = self.model.training
                if training:
                    self.model.eval()
                o = self.model(torch.zeros(1, in_chans, img_size, img_size))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                self.num_patches = o.shape[-2] * o.shape[-1]
                feature_dim = o.shape[1]
                self.model.train(training)
        else:
            self.num_patches = feature_size * feature_size
            feature_dim = model.features[-1].num_features

        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, multiview_images):
        """
            Map input images to features
        :param multiview_images: Input Images, shape: [B, N, 3, H (224), W (224)]
        :return: Features, shape: [B, N, G, D]
        """
        multiview_imgs = multiview_images.permute(1, 0, 2, 3, 4).contiguous()
        multiview_imgs = torch.split(multiview_imgs, 1, dim=0)
        all_img_features = []

        for singleview_img in multiview_imgs:
            single_img_features = self.proj(self.model(singleview_img.squeeze(dim=0)))  # shape => [B, D, h, w]
            all_img_features.append(single_img_features)

        all_img_features = torch.stack(all_img_features)  # shape => [B, N, D, h, w]
        all_img_features = all_img_features.permute(1, 0, 3, 4, 2).contiguous()  # shape => [B, N, h, w, D]
        all_img_features = all_img_features.reshape(all_img_features.shape[0], all_img_features.shape[1],
                                                    -1, all_img_features.shape[-1])  # shape => [B, N, hw, D]

        return all_img_features

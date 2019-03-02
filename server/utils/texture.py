import cv2
import numpy as np
import skimage.io as io
from skimage.measure import label


class Texture:
    def __init__(self,config):
        self.mode='read_from_file'
        self.texture_path=config['texture_img']
        self.Grid_Pixels=config['grid_pixels']
        self.read_texture()
        self.config=config
    def read_texture(self):
        texture_img=cv2.imread(self.texture_path)[:,:,::-1]/255.
        self.TextureIm = np.zeros([24, self.Grid_Pixels, self.Grid_Pixels, 3])
        for i in range(4):
            for j in range(6):
                self.TextureIm[(6 * i + j), :, :, :] = \
                    texture_img[(self.Grid_Pixels * j):(self.Grid_Pixels * j + self.Grid_Pixels),
                    (self.Grid_Pixels * i):(self.Grid_Pixels * i + self.Grid_Pixels), :]

    def transfer_texture(self, im, IUV):
        TextureIm = self.TextureIm
        generated_image = im.copy()
        for PartInd in range(1, 23):  ## Set to xrange(1,23) to ignore the face part.
            tex = TextureIm[PartInd - 1, :, :, :].squeeze()  # get texture for each part.
            u_current_points = IUV[..., 1][IUV[:, :, 0] == PartInd]  # Pixels that belong to this specific part.
            v_current_points = IUV[..., 2][IUV[:, :, 0] == PartInd]
            mask = ((255 - v_current_points) * self.config['uv_dim'] / 255.).astype(int), (
                    u_current_points * self.config['uv_dim'] / 255.).astype(int)
            tex_to_rep = tex[mask][..., ::-1]
            generated_image[IUV[:, :, 0] == PartInd] = (tex_to_rep * 255).astype(np.uint8)
        return generated_image

    def parse_individuals(self,iuv,im,area_thresh=3500):
        i=iuv[...,0].copy()
        i[i>0]=1
        labelled_i,num=label(i,return_num=True)
        crop_coords=[]
        for i in range(1,num+1):
            mask=labelled_i == i
            x,y=np.where(mask)
            area=x.shape[0]
            if (area<area_thresh):
                labelled_i[mask]=0
            else:
                coord=(i,np.min(x),np.min(y),np.max(x),np.max(y))
                crop_coords.append(coord)
        unique=np.unique(labelled_i)
        cropped_images=[]
        cropped_iuvs = []
        for i in range(unique.shape[0]-1):
            xmin=crop_coords[i][1]
            ymin=crop_coords[i][2]
            xmax=crop_coords[i][3]
            ymax=crop_coords[i][4]
            cropped_images.append(im[xmin:xmax,ymin:ymax])
            cropped_iuvs.append(iuv[xmin:xmax,ymin:ymax])
        return cropped_images,cropped_iuvs

    def get_individual_texture(self,im,IUV,TextureIm):
        for PartInd in range(1, 25):  ## Set to xrange(1,23) to ignore the face part.
            u_current_points = IUV[..., 1][IUV[:, :, 0] == PartInd]  # Pixels that belong to this specific part.
            v_current_points = IUV[..., 2][IUV[:, :, 0] == PartInd]
            mask = ((255 - v_current_points) * self.config['uv_dim'] / 255.).astype(int), (
                    u_current_points * self.config['uv_dim'] / 255.).astype(int)

            texture_mask = np.sum(TextureIm[PartInd - 1, mask[0], mask[1], ::-1], axis=-1) == 0
            texture_mask = texture_mask[..., np.newaxis]
            TextureIm[PartInd - 1, mask[0], mask[1], ::-1]+=im[IUV[:, :, 0] == PartInd]*texture_mask
        return TextureIm

    def save_texture(self,texture,name):
        texture_img=np.zeros((6*self.Grid_Pixels,4*self.Grid_Pixels,3))
        for i in range(4):
            for j in range(6):
                    current_texture=texture[(6 * i + j)]
                    r_t=current_texture[...,0]
                    g_t=current_texture[...,1]
                    b_t=current_texture[...,2]
                    mask=np.sum(current_texture,axis=-1)!=0
                    mean_r=np.mean(r_t[mask])
                    mean_g=np.mean(g_t[mask])
                    mean_b=np.mean(b_t[mask])
                    current_texture[~mask]=np.array([mean_r,mean_g,mean_b])
                    texture_img[(self.Grid_Pixels * j):(self.Grid_Pixels * j + self.Grid_Pixels),
                    (self.Grid_Pixels * i):(self.Grid_Pixels * i + self.Grid_Pixels), :]=\
                        current_texture
        io.imsave(name,texture_img.astype(np.uint8))
        return texture_img.astype(np.uint8)

    def extract_multiple_textures(self,iuv,im,name):
        individuals,iuvs=self.parse_individuals(iuv,im)
        for img_idx in range(len(individuals)):
            texture=self.get_individual_texture(individuals[img_idx],iuvs[img_idx],np.zeros(self.TextureIm.shape))
            self.save_texture(texture,(name+str(img_idx)+'.jpg'))

    def extract_texture_from_video(self,img_frames,iuv_frames,name):
        texture= np.zeros(self.TextureIm.shape)
        for i in range(len(img_frames)):
            self.get_individual_texture(img_frames[i],iuv_frames[i],texture)
        self.save_texture(texture,name)
    def transfer_texture_on_video(self,images,iuvs):
        out = []
        for i in range(len(images)):
            out.append(self.transfer_texture(images[i], iuvs[i]))
        return out

clear all; close all; clc;

load('results_model/result_fullres_mouseclick_ours_noalphaloss.mat');
Radius = 20;

ImageChosen = [38]; %the 38th image
ImageChosenMapInd = [1]; %for test trial only (the first trial)
typelist = [1:8];
imgsize = 400;
figure;

load('Mat/Categories_GT_573.mat');
for i = 1:length(ImageChosen)
    %load(['/home/mengmi/Projects/Proj_context1/pytorch/recurrentVA_noalphaloss/results/trial_' num2str(ImageChosen(i)) '.mat']);
    load(['../src/results/trial_' num2str(1) '.mat']);
    
    if cateinfor{ImageChosen(i),1}~= (predicted_seq(9)+1)
        continue;
    end
    
    for t = 1:length(typelist)
        clickimg = squeeze(clickS(1, typelist(t)+1,:,:,:));
        clickimg = uint8(clickimg);
        clickimg = imresize(clickimg,[imgsize imgsize]);        
    
        clickplot = clickimg;
        for s = 1:t
            pos = [int32(Modelclicks( ImageChosenMapInd(i)).mouseclick(1,s)/1280*imgsize); int32(Modelclicks(ImageChosenMapInd(i)).mouseclick(2,s)/1024*imgsize); Radius]';
            clickplot = insertShape(clickplot,'FilledCircle',pos,'Color','red','LineWidth',5);
          
        end
        
        clickplot = imresize(clickplot, [imgsize imgsize]);
        subplot(1,2,1);
        imshow(clickplot);
        title(['The ' num2str(t) 'th click (red dot)']);
        %imwrite(clickplot, ['ICLR_plotsimages_rebuttal/img_' num2str(i) '_' num2str(t) '.png']);
        %Modelclicks(j).mouseclick(1:t);
        
        clickimg = imresize(clickimg, [imgsize imgsize]);
        
        attentionmap = reshape(alphas(t,:), [sqrt(size(alphas,2)), sqrt(size(alphas,2))]);
        attentionmap = attentionmap';
        attentionmap = mat2gray(attentionmap);
        attentionmap = imresize(attentionmap, [imgsize imgsize]);
        hsize = [20 20];
        sigma = 5;
        H = fspecial('gaussian',hsize,sigma);
        attentionmap = imfilter(attentionmap,H,'replicate');
        attentionmap = mat2gray(attentionmap);
        
        heat = heatmap_overlay(clickimg,attentionmap);
        %imwrite(clickimg, ['ICLR_plotsimages_rebuttal/attention_' num2str(i) '_' num2str(t) '.png'], 'png', 'Alpha', attentionmap );
        %imwrite(attentionmap, ['ICLR_plotsimages_rebuttal/attend_' num2str(i) '_' num2str(t) '.png']);
        %I = imread(['ICLR_plotsimages_rebuttal/attention_' num2str(i) '_' num2str(t) '.png']);
        str= nms{predicted_seq(t)+1};
        position = [1 1];
        I = insertText(attentionmap,position,str,'FontSize',55,'BoxColor','red','BoxOpacity',0.4,'TextColor','white');
        
        subplot(1,2,2);
        imshow(I);
        title(['Attention after ' num2str(t-1) 'th click']);
        %imwrite(I, ['ICLR_plotsimages_rebuttal/attention_' num2str(i) '_' num2str(t) '.png']);
        pause;
    end
    
end
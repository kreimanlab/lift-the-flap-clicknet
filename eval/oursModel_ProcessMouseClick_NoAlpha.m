clear all; close all; clc;

load('Mat/Categories_GT_573.mat');
clicksteps = 9;
totalNumImg = 573;
GPtest = [];
width = 1024; height = 1280;
Modelclicks = [];

for i = 38 %1:totalNumImg %modify this for total 573 images evaluation; currently only evaluating the 38th image
    i
    load(['../src/results/trial_' num2str(1) '.mat']);
    %load(['/home/mengmi/Projects/Proj_context1/pytorch/recurrentVA_noalphaloss/results/trial_' num2str(i) '.mat']);
    clickx = []; clicky = [];
    trial = [];
    
    for t = 1:clicksteps
        % attention map
        attentionmap = reshape(alphas(t,:), [sqrt(size(alphas,2)), sqrt(size(alphas,2))]);
        attentionmap = attentionmap';
        
        attentionmap = mat2gray(attentionmap);
        attentionmap = imresize(attentionmap, [width height]);
        
        maximum = max(max(attentionmap));
        [row_y,col_x]=find(attentionmap==maximum); %x is horizontal; y is vertical
        clickx = [clickx col_x(1)]; %extract click coordinate
        clicky = [clicky row_y(1)];
 
    end
    
%     clickx = int32(clickx/sqrt(size(alphas,2))*height);
%     clicky = int32(clicky/sqrt(size(alphas,2))*width);
%     
%     clickx(find(clickx<1)) = 1;
%     clicky(find(clicky<1)) = 1;
%     clickx(find(clickx>height)) = height;
%     clicky(find(clicky>width)) = width;
    
%     testmask = zeros(width, height,3);
%     testmask = insertShape(attentionmap,'FilledCircle',[ clickx(9) clicky(9) 100],'Color','white');
%     imshow(testmask);
    
    trial.mouseclick = [clickx; clicky];
    Modelclicks = [Modelclicks trial];
    GPtest = [GPtest; predicted_seq'];
end
GPtest = GPtest + 1;

correctlist = [];
for i = 38 %1:totalNumImg %modify this for total 573 images evaluation; currently only evaluating the 38th image
    gt = cateinfor{i,1};
    correct = (GPtest(1,:) == gt);
    correctlist = [correctlist; correct];
end

resultModel = correctlist;
save('results_model/result_fullres_mouseclick_ours_noalphaloss.mat','resultModel','Modelclicks');
display(['inference accu = ' num2str(nanmean(correctlist))]);


% for type = 1: 4 %different types of mouse clicks: 1 = 1 click; 2 = 2 clicks; 3 = 4 clicks; 4 = 8 clicks
%     convertedTotalBin ={};
%     
%     if type == 1
%         ind = 2;
%     elseif type == 2
%         ind = 3;
%     elseif type == 3
%         ind = 5;
%     else
%         ind = 9;
%     end
%     
%     for i = 1:totalNumImg
%         vec = correctlist( i, ind);
%         convertedTotalBin{i} = vec;
%         
%     end
%     save(['results_model/ours_result_colorimg_MSCOCO_type_' num2str(type) '_noalphaloss.mat'],'convertedTotalBin');
% end
    
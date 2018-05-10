I = imread('./best.png');
I = rgb2gray(I);
imshow(I);
figure;
imcontour(I,1);
[J, rect] = imcrop(I);
imwrite(J,'./cont.png');
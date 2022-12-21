function Hist_IQH_4D = Fun_2DHistCNN(Bins_no,Signal_I,Signal_Q)

   %Number of Bins
Value_Rng=4;
Edge=linspace(-Value_Rng,Value_Rng,Bins_no+1);

[Raw Col]=size(Signal_I)
for i=1:Col  
  TempH=histcounts2(Signal_I(:,i),Signal_Q(:,i),Edge,Edge);
  histogram2(Signal_I(:,101),Signal_Q(:,101),Edge,Edge);
  Hist_IQH_4D(:,:,1,i) = TempH;
end


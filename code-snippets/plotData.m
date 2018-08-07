function [XY,cur,err,RT]=plotData(ID)
%% [XY,cur,err,RT]=plotData(ID)
% XY: tensor data for the trajectory of cursor
% cur: cursor position in degree, relative to the top, CW: -, CCW: +
% err: error between cursor and target positions, CW: -, CCW: +
% RT: Reaction Time (ms)
%
% Date: 12/05/2017
% Written by Sungshin Kim
%
[XY,cur,err,rot,col,RT,nTr]= calcTraj(ID);
figure; 
idx0 = find(col==0);idx1=find(col==1 & rot==1);idx2 = find(col==2 & rot==2); 
idx3 = find(col==1 & rot==0); idx4 = find(col==2 & rot==0);
plot(idx0,cur(idx0),'y.','Markersize',25);
hold on;
plot(idx1,30-cur(idx1),'g.','Markersize',25);
plot(idx2,-30-cur(idx2),'b.','Markersize',25);
plot(idx3,-cur(idx3),'g.','Markersize',25);
plot(idx4,-cur(idx4),'b.','Markersize',25);
ylim([-50 50]);
vline(nTr+1,'k:');
hline([-30 30],'k:');
xlabel('Trials','Fontsize',40);
ylabel('Direction (deg)','Fontsize',40);
title(ID,'Fontsize',50);
set(gca,'Box','Off','Fontsize',20);
set(gcf,'Color','W');



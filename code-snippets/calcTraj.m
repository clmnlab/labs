function [XY,cur,err,mov,rot,col,RT,nTr] = calcTraj(ID)
%% [XY,cur,err,rot,col,RT,nS]=calcTraj(ID)
% XY: tensor data for the trajectory of cursor
% cur: cursor position in degree, relative to the top, CW: -, CCW: +
% err: error between cursor and target positions, CW: -, CCW: +
% rot: rotation, 0: No rotation, 1: task 1, -30 (CW), 2: task 2, +30 (CCW)
% col: color code, 0: control, 1: task 1, 2: task 2
% RT: Reaction Time (ms)
% nTr: Number of trials for each run
%
% Date: 12/05/2017
% Written by Sungshin Kim
%
dist = 350; 
traj = importdata([ID '_XY.txt']);
Tr = traj.data(:,1);
N = length(unique(Tr));
tar = importdata([ID '_XYii.txt']);
tem = importdata([ID '_study.txt']);
RT = tem.data(:,6);
nTr = [cumsum(diff(find(tem.data(:,7)<5000))); length(tem.data)];
% nTr = 80;
rot = tem.data(:,3);
col = tem.data(:,5);
n = 1;
trials = tem.data(:,1);

for n=1:length(trials)
        idx = find(Tr==trials(n));
        X = traj.data(idx,2); Y = traj.data(idx,3);
        XY(1,n,:) = X;
        XY(2,n,:) = Y;
    if RT(n)~=1500
        tpos(n,:) = tar.data(n,[6 7]);
        xf = X(end); yf = Y(end);
        err(n) = calc_err(xf,yf,tpos(n,:));
        cur(n) = calc_err(xf,yf,[-dist 0]);
    else
        err(n) = NaN;
        err(n) = NaN; 
        cur(n) = NaN;
    end
end
if strcmp(ID,'HAK20180109')
    col(find(col==3))=0; rot(find(rot==3))=0;
end;
idx0 = find(col==0);idx1=find(col==1 & rot==1);idx2 = find(col==2 & rot==2); 
idx3 = find(col==1 & rot==0); idx4 = find(col==2 & rot==0);
mov = zeros(1,length(err));
mov(idx0) = cur(idx0); mov(idx1) = 30-cur(idx1); mov(idx2) = -30-cur(idx2);
mov(idx3) = -cur(idx3); mov(idx4) = -cur(idx4);



function err = calc_err(xf,yf,tpos)
xyTheta = cart2pol(xf,yf)*180/pi;
if xyTheta < 0 xyTheta = xyTheta +360; end;
tarTheta = cart2pol(tpos(1),tpos(2))*180/pi;
if tarTheta < 0 tarTheta = tarTheta +360; end;
err = xyTheta - tarTheta; 
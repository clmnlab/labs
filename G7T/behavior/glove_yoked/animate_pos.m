function animate_pos(id_n,session_n, trial_n, animate_time)
    close all
    folname='main_ex';
    subID=strcat(folname,strcat('/GT_Main_',num2str(id_n),'/GT_Main_',num2str(id_n)));

    switch session_n
        case 1
            inputsession=load(strcat(subID,'_1.mat'));
        case 2
            inputsession=load(strcat(subID,'_2.mat'));
        case 3
            inputsession=load(strcat(subID,'_3.mat'));
        case 4
            inputsession=load(strcat(subID,'_4.mat'));
        case 5
            inputsession=load(strcat(subID,'_5.mat'));
        case 6
            inputsession=load(strcat(subID,'_6.mat'));
    end
    
    stim1_pos=inputsession.positionset1;
    stim2_pos=inputsession.positionset2;
    xSize=inputsession.xCenter*2;
    ySize=inputsession.yCenter*2;
    test_seq=inputsession.test_type;
    
    switch test_seq(trial_n)
        case 1
            stimPos=stim1_pos;
        case 2
            stimPos=stim2_pos;
    end

    XY=inputsession.total_XY{trial_n};
    radius= 100;
    axis([0 xSize -ySize 0]);
    hold on
    for i = 1:5
        circle(stimPos(1,i),-stimPos(2,i),radius);
    end

    curve = animatedline(); 
    for j = 1:length(XY)
        addpoints(curve,XY(1,j),-XY(2,j))
        drawnow()
        pause(animate_time)
    end

    hold off
end
function h = circle(x,y,r)
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = plot(xunit, yunit,'b');
end

folname='main_ex';
ID_n=3;
subID=strcat(folname,'/GT_Main_',num2str(ID_n),'/GT_Main_',num2str(ID_n));
sessions=load_session(subID,6);
inputtrial=1;
inputsession=sessions{1};
totalcoord=inputsession.total_XY{inputtrial};
pre_v=totalcoord(:,2)-totalcoord(:,1);
post_v=totalcoord(:,3)-totalcoord(:,2);
angle_v=angleV(pre_v,post_v);



function sessions=load_session(subID,total_n)
    sessions=cell(1,total_n);
    for i = 1:total_n
        sessions{i}=load(strcat(subID,'_',num2str(i),'.mat'));
    end
end

function cosangle=angleV(a,b)
   cosangle=rad2deg(acos((a*b')/(norm(a)*norm(b))));
end
function [] = my_depictClasses(K1,K2,K3,K4)
    figure
    hold all
    title('Klase')
    scatter(K1(1,:),K1(2,:),'m*');
    scatter(K2(1,:),K2(2,:),'bo');
    scatter(K3(1,:),K3(2,:),'g.');
    scatter(K4(1,:),K4(2,:),'x','Color',[0.5 0.75 0.25]);
    grid on
    grid minor
    xlabel('x1');
    ylabel('x2');
    legend('K1','K2','K3','K4');
end
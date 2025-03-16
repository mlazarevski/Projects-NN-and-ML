%% Parametri i graficki prikaz
clear; close all; clc;

N = 500; P1 = 0.4; P2 = 0.8;
M1 = [0 0]';
S1 = [1 0.5; 0.5 1];
M2 = [6 1]';
S2 = [1.5 -0.7; -0.7 1.5];
M3 = [6 6]';
S3 = [1 0.6; 0.6 1];
M4 = [12 2]';
S4 = [2 0.2; 0.2 2];

K11 = mvnrnd(M1,S1,N)';
K12 = mvnrnd(M2,S2,N)';
K21 = mvnrnd(M3,S3,N)';
K22 = mvnrnd(M4,S4,N)';

pom = rand(1,N); 
K1 = (pom<=P1).*K11 + (pom>P1).*K12;
K2 = (pom<=P2).*K21 + (pom>P2).*K22;

figure(1)
hold all
title('Bimodalne klase')
scatter(K1(1,:),K1(2,:),'r*');
scatter(K2(1,:),K2(2,:),'bo');
grid on
grid minor
xlabel('x1');
ylabel('x2');
legend('K1','K2');

M1_est = mean(K1')';
S1_est = cov(K1');
M2_est = mean(K2')';
S2_est = cov(K2');

%% Racunanje fgv i histograma
step = 0.1;
x = -5:step:20;
y = -5:step:10;

f1 = zeros(length(x),length(y));
f2 = f1; h = f1;
for i=1:length(x)
    for j=1:length(y)
        X = [x(i); y(j)];
        pdf_values1 = 1 / (2 * pi * sqrt(det(S1))) * exp(-0.5 * (X - M1)'*inv(S1)*(X - M1));
        pdf_values2 = 1 / (2 * pi * sqrt(det(S2))) * exp(-0.5 * (X - M2)'*inv(S2)*(X - M2));
        f1(i,j) = P1*pdf_values1 + (1-P1)*pdf_values2;
        pdf_values3 = 1 / (2 * pi * sqrt(det(S3))) * exp(-0.5 * (X - M3)'*inv(S3)*(X - M3));
        pdf_values4 = 1 / (2 * pi * sqrt(det(S4))) * exp(-0.5 * (X - M4)'*inv(S4)*(X - M4));
        f2(i,j) = P2*pdf_values3 + (1-P2)*pdf_values4;
        
        h(i,j) = - log(f1(i,j)) + log(f2(i,j));
    end
end

f1 = f1/sum(sum(f1));
f2 = f2/sum(sum(f2));

figure
sgtitle('Klasa K2')
subplot(211)
hist3(K2'); 
title('Histogram')
subplot(212)
surf(x, y, f2', 'EdgeColor', 'none');
title('Funkcija gustine verovatnoće')

figure
sgtitle('Klasa K1')
subplot(211)
hist3(K1'); 
title('Histogram')
subplot(212)
surf(x, y, f1', 'EdgeColor', 'none');
title('Funkcija gustine verovatnoće') 

%% Bayes klasifikator
% N1 = N2 -> P1 = P2 = 1/2
% -ln(P2/P1) = 1

figure(10)
hold all
title('Bayes klasifikator')
scatter(K1(1,:),K1(2,:),'r*');
scatter(K2(1,:),K2(2,:),'bo');
contour(x,y,h',[0 0])
grid on
grid minor
xlabel('x1');
ylabel('x2');
legend('K1','K2','Bayes');

figure(9)
hold all
title('Klasifikatori za testiranje hipoteza')
scatter(K1(1,:),K1(2,:),'r*');
scatter(K2(1,:),K2(2,:),'bo');
contour(x,y,h',[0 0])
grid on
grid minor
xlabel('x1');
ylabel('x2');
legend('K1','K2','Bayes');

%% Greska
e_metod1 = [sum(f1(h > 0)); sum(f2(h < 0))].*step^2;

for i=1:length(x)
    for j=1:length(y)
        X = [x(i); y(j)];
        pdf_values1 = 1 / (2 * pi * sqrt(det(S1))) * exp(-0.5 * (X - M1)'*inv(S1)*(X - M1));
        pdf_values2 = 1 / (2 * pi * sqrt(det(S2))) * exp(-0.5 * (X - M2)'*inv(S2)*(X - M2));
        f1(i,j) = P1*pdf_values1 + (1-P1)*pdf_values2;
        pdf_values3 = 1 / (2 * pi * sqrt(det(S3))) * exp(-0.5 * (X - M3)'*inv(S3)*(X - M3));
        pdf_values4 = 1 / (2 * pi * sqrt(det(S4))) * exp(-0.5 * (X - M4)'*inv(S4)*(X - M4));
        f2(i,j) = P2*pdf_values3 + (1-P2)*pdf_values4;
        
        h(i,j) = - log(f1(i,j)) + log(f2(i,j));
        
    end
end


e_metod2 = [0;0];

for i=1:length(K1)
    X = K1(:,i);
    pdf_values1 = 1 / (2 * pi * sqrt(det(S1))) * exp(-0.5 * (X - M1)'*inv(S1)*(X - M1));
    pdf_values2 = (1 / (2 * pi * sqrt(det(S2)))) * exp(-0.5 * (X - M2)'*inv(S2)*(X - M2));
    f1_temp = P1*pdf_values1 + (1-P1)*pdf_values2;
    pdf_values3 = (1 / (2 * pi * sqrt(det(S3)))) * exp(-0.5 * (X - M3)'*inv(S3)*(X - M3));
    pdf_values4 = (1 / (2 * pi * sqrt(det(S4)))) * exp(-0.5 * (X - M4)'*inv(S4)*(X - M4));
    f2_temp = P2*pdf_values3 + (1-P2)*pdf_values4;
    
    H1(i) = - log(f1_temp) + log(f2_temp);
    e_metod2(1) = e_metod2(1) + 1 * (0 < H1(i));
end

for i=1:length(K2)
    X = K2(:,i);
    pdf_values1 = 1 / (2 * pi * sqrt(det(S1))) * exp(-0.5 * (X - M1)'*inv(S1)*(X - M1));
    pdf_values2 = (1 / (2 * pi * sqrt(det(S2)))) * exp(-0.5 * (X - M2)'*inv(S2)*(X - M2));
    f1_temp = P1*pdf_values1 + (1-P1)*pdf_values2;
    pdf_values3 = (1 / (2 * pi * sqrt(det(S3)))) * exp(-0.5 * (X - M3)'*inv(S3)*(X - M3));
    pdf_values4 = (1 / (2 * pi * sqrt(det(S4)))) * exp(-0.5 * (X - M4)'*inv(S4)*(X - M4));
    f2_temp = P2*pdf_values3 + (1-P2)*pdf_values4;
    
    H2(i) = - log(f1_temp) + log(f2_temp); % Wald's Sequential Test
    e_metod2(2) = e_metod2(2) + 1 * (0 > H2(i));
end
e_metod2 = e_metod2/N;

display(sprintf(['Metod:\t' 'Greška prvog tipa' '\tGreška drugog tipa']))
display(['Teorijski:  ' num2str(e_metod1')])
display(['Empirijski: ' num2str(e_metod2')])
%% Test minimalne cene
C11 = 0; C22 = 0;
C21 = 5; C12 = 1;
T = - log((C12 - C22) / (C21 - C11)) ;

figure(9)
contour(x,y,h',[T T], 'm--')
legend('K1','K2','Bayes','Test minimalne cene');

%% Constant False Alarm Ratio
% Neuman-Pearson test
% e2 = e0 jer je e0 vec minimalna greska, nadamo se da ce ovaj uslov imati
% posledicu e1 -> 0

e0 = sum(e_metod1);
e2 = e0;

Nmi = 100*N;
pom = rand(1,Nmi); 
K2_temp = (pom<=P2).*mvnrnd(M3,S3,Nmi)' + (pom>P2).*mvnrnd(M4,S4,Nmi)';

for i=1:Nmi
    X = K2_temp(:,i);
    pdf_values1 = 1 / (2 * pi * sqrt(det(S1))) * exp(-0.5 * (X - M1)'*inv(S1)*(X - M1));
    pdf_values2 = 1 / (2 * pi * sqrt(det(S2))) * exp(-0.5 * (X - M2)'*inv(S2)*(X - M2));
    f1_mi = P1*pdf_values1 + (1-P1)*pdf_values2;
    pdf_values3 = 1 / (2 * pi * sqrt(det(S3))) * exp(-0.5 * (X - M3)'*inv(S3)*(X - M3));
    pdf_values4 = 1 / (2 * pi * sqrt(det(S4))) * exp(-0.5 * (X - M4)'*inv(S4)*(X - M4));
    f2_mi = P2*pdf_values3 + (1-P2)*pdf_values4;
    
    h_mi(i) = - log(f1_mi) + log(f2_mi);
end


h_mi = sort(h_mi);
mi = exp(-h_mi(round(e0*Nmi)));

T = log(mi);

figure(9)
contour(x,y,h',[T T], 'g:')
legend('K1','K2','Bayes','Test minimalne cene','CFAR');

%% Wald
e = [1e-100, 1e-50];
a = -log ((1 - e(1))/e(2)); b = -log(e(1)/(1-e(2)));

seq1_len = [];
seq2_len = [];
figure; hold all;
for i = 1:100
    [Seq1, Seq2] = my_wald(H1(randperm(N)), H2(randperm(N)), a, b);

    plot(Seq1, 'r');
    plot(Seq2, 'b');
    
    seq1_len = [seq1_len length(Seq1)];
    seq2_len = [seq2_len length(Seq2)];
end

plot([0, max([seq1_len seq2_len])], [a, a], 'k--'); 
plot([0, max([seq1_len seq2_len])], [b, b], 'k--');

hold off;
xlabel('Broj odbiraka');
title('Sekvencijalni Wald test');
legend('K1', 'K2', 'Location', 'best');

Ehw1 = sum(sum(step^2*f1.*h));
Ehw2 = sum(sum(step^2*f2.*h));

Esw1 = a*(1-e(1))+b*e(1);
Esw2 = b*(1-e(2))+a*e(2);

Em1 = Esw1/Ehw1; Em2 = Esw2/Ehw2;

display('Teorijski')
display(sprintf('Srednja sekvenca w1: %d\nSrednja sekvenca w2: %d', Em1, Em2));
display('Empirijski')
display(sprintf('Srednja sekvenca w1: %d\nSrednja sekvenca w2: %d', mean(seq1_len), mean(seq2_len)));

%% Broj potrebnih odbiraka Waldovog testa u zavisnosti od greške prvog i drugog tipa

E1 = logspace(-1, -100, 100); E2 = logspace(-1, -100, 100); 

Em1 = zeros(size(E1)); Em2 = zeros(size(E2));

cnt1 = 0;
for e1 = E1
    cnt1 = cnt1 + 1;
    cnt2 = 0;
    for e2 = E2
        cnt2 = cnt2 + 1;
        a = -log ((1 - e(1))/e(2)); b = -log(e(1)/(1-e(2)));

        Ehw1 = sum(sum(step^2*f1.*h));
        Ehw2 = sum(sum(step^2*f2.*h));
        
        Esw1 = a*(1-e1)+b*e1;
        Esw2 = b*(1-e2)+a*e2;

        Em1(cnt1, cnt2) = Esw1/Ehw1; Em2(cnt1, cnt2) = Esw2/Ehw2;
    end
end

%%
figure
sgtitle('Uslovno očekivanje za neophodan broj odbiraka')
subplot(1, 2, 1);
surf(E1, E2, Em1, 'EdgeColor', 'none');
xlabel('\epsilon_1');
ylabel('\epsilon_2');
zlabel('$E\{m|\omega\}$', 'Interpreter', 'latex');
title('K1');

set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');

subplot(1, 2, 2);
surf(E1, E2, Em2, 'EdgeColor', 'none');
xlabel('\epsilon_1');
ylabel('\epsilon_2');
zlabel('$E\{m|\omega\}$', 'Interpreter', 'latex');
title('K2');

set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');




%% Pojedinačni Wald test za ceo skup
function [Seq1, Seq2] = my_wald(H1, H2, a, b)
    Sm = 0; Seq1 = [];
    for h1 = H1
        Sm = Sm + h1;
        Seq1 = [Seq1 Sm];
        if Sm < a
            break
        end
    end
    Sm = 0; Seq2 = [];
    for h2 = H2
        Sm = Sm + h2;
        Seq2 = [Seq2 Sm];
        if Sm > b
            break
        end
    end
end




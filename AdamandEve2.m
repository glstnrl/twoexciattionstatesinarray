clc;
clear all;
close all;

%S=((sin(teta)*cos(-r/c))/r)^2;
%E=((sin(teta)*cos(-r/c))/r)^2 theta;
%B=((sin(teta)*cos(-r/c))/r)^2 phi;
lamda=6*pi*10^(-6);
R=100*lamda;
k=(2*pi)/lamda;
C=1.4*10^(-19);
Evec=0;
Bvec=0;
dteta=pi/4;
dphi=0;
dvec=[sin(dteta)*cos(dphi),sin(dteta)*sin(dphi),cos(dteta)];

hbar=1.054571817e-34;
eps=8.8541878128e-12;
c=2.99792458e8;
mu=2.54e-29;
N=11;
w=k*c;
rvec=[1,0,0];
mudotr=dot(rvec,dvec);
modofsystem=5;

H=zeros(N,N);
Omega=zeros(N,N);
Gamma=zeros(N,N);
r=zeros(N,N);

 for i=1:N     
    for j=1:N 
        ri=0.4*lamda*i;
        rj=0.4*lamda*j;
        r(i,j)=abs(rj-ri);
        if i==j
            Gamma(i,j)=((w^3)*(mu^2))/(3*pi*eps*hbar*(c)^3);
        end
            if i~=j  
            Omega(i,j)=(3/4)*((w^3)*(mu^2))/(3*pi*eps*hbar*(c)^3)*(((-(1-mudotr^2)*cos(k*r(i,j)))/(k*r(i,j)))+((1-3*mudotr^2)*((sin(k*r(i,j)))/(k*r(i,j))^2+cos(k*r(i,j))/(k*r(i,j))^3)));
            Gamma(i,j)=(3/2)*((w^3)*(mu^2))/(3*pi*eps*hbar*(c)^3)*((1-mudotr^2)*(sin(k*r(i,j))/(k*r(i,j)))+(1-3*mudotr^2)*(cos(k*r(i,j))/(k*r(i,j))^2-sin(k*r(i,j))/(k*r(i,j))^3));
            end
      end
end

H=Omega-(1i/2)*Gamma;
[dipolemagarr,Energy]=eig(H);
dipolemag=dipolemagarr(:,modofsystem);
nofdmag=N/2+0.5;
flag=true;

i=1;
for teta=0:0.01:pi    
    for phi=0:0.01:2*pi
        posS=[R*sin(teta)*cos(phi),R*sin(teta)*sin(phi),R*cos(teta)];
            absR1=posS-[0,0,0];
            d1=sqrt(sum(absR1.^2));
            teta1=acos(absR1(3)/d1);
            phi1=atan2(absR1(2),absR1(1));
            sinterm1=norm(cross(absR1,dvec))/d1;
            q0=abs(dipolemag(nofdmag));
            pha=angle(dipolemag(nofdmag));
            E1=q0*((sinterm1*cos(-d1*k+pha))/d1);
            B1=q0*((sinterm1*cos(-d1*k+pha))/d1);
            E1vec=[E1*cos(teta1)*cos(phi1),E1*cos(teta1)*sin(phi1),-E1*sin(teta1)];
            B1vec=[-B1*sin(phi1),B1*cos(phi1),0];
            Evec=Evec+E1vec;
            Bvec=Bvec+B1vec;
        for d=0.4*lamda:0.4*lamda:lamda*2
            posS=[R*sin(teta)*cos(phi),R*sin(teta)*sin(phi),R*cos(teta)];
            absR1=posS-[d/2,0,0];
            absR2=posS-[-d/2,0,0];
            d1=sqrt(sum(absR1.^2));
            d2=sqrt(sum(absR2.^2));
            teta1=acos(absR1(3)/d1);
            teta2=acos(absR2(3)/d2);
            phi1=atan2(absR1(2),absR1(1));
            phi2=atan2(absR2(2),absR2(1));
            sinterm1=norm(cross(absR1,dvec))/d1;
            sinterm2=norm(cross(absR2,dvec))/d2;
            nofdmag1=nofdmag+d/(0.4*lamda);
            nofdmag2=nofdmag-d/(0.4*lamda);
            q1=abs(dipolemag(nofdmag1));
            q2=abs(dipolemag(nofdmag2));
            pha1=angle(dipolemag(nofdmag1));
            pha2=angle(dipolemag(nofdmag2));
            E1=q1*((sinterm1*cos(-d1*k+pha1))/d1);
            E2=q2*((sinterm2*cos(-d2*k+pha2))/d2);
            B1=q1*((sinterm1*cos(-d1*k+pha1))/d1);
            B2=q2*((sinterm2*cos(-d2*k+pha2))/d2);
            E1vec=[E1*cos(teta1)*cos(phi1),E1*cos(teta1)*sin(phi1),-E1*sin(teta1)];
            E2vec=[E2*cos(teta2)*cos(phi2),E2*cos(teta2)*sin(phi2),-E2*sin(teta2)];
            B1vec=[-B1*sin(phi1),B1*cos(phi1),0];
            B2vec=[-B2*sin(phi2),B2*cos(phi2),0];
            Evec=Evec+E1vec+E2vec;
            Bvec=Bvec+B1vec+B2vec;
            Svec=cross(Evec,Bvec);
        end
        S=C*sqrt(sum(Svec.^2));
        
            Sshow(i)=S;
            tetavec(i)=teta;
            phivec(i)=phi;
            i=i+1;  
        
        Evec=0;
        Bvec=0;
    end
end

x=R*sin(tetavec).*cos(phivec);
y=R*sin(tetavec).*sin(phivec);
z=R*cos(tetavec);

figure;
scatter3(x,y,z,15,log10(Sshow),'filled');
colorbar;
axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title('Radiation Scale');
grid on;
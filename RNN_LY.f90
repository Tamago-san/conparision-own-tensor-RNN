!-----------------------------------------------------------------------------
!ローレンツモデルのリアプノフ指数を求める。
!
!モデルのタイムスケールの0.01で進む。
!gfortran RNN_LY_lynoyatu.f90 rnn_tanh_lynoyatu.f90 -o RNN_LY.out -llapack -lblas
!-----------------------------------------------------------------------------
module lyapnov
    implicit none
    integer, parameter :: rc_node = 10
    integer, parameter :: par_node = 3
!    real(8), parameter :: par_a = 16.d0
!    real(8), parameter :: par_b = 40.d0
!    real(8), parameter :: par_c = 4.d0
    real(8), parameter :: par_a = 10.d0
    real(8), parameter :: par_b = 28.d0
    real(8), parameter :: par_c = 8.d0/3.d0
    real(8), parameter  :: dt=1.d-2
    real(8), parameter :: dt_Runge = 1.d-4
    real(8), parameter :: dx0 = 1.d0
    real(8), parameter :: dy0 = 4.d0
    real(8), parameter :: dz0 = 2.d0
    real(8), parameter :: x0 = 0.d0
    real(8), parameter :: y0 = 0.1d0
    real(8), parameter :: z0 = 0.d0
    integer, parameter :: lyapnov_step = 2000 !リヤプノフ指数計測回数
    integer, parameter :: tau =20
    integer, parameter :: rc_step = 1000
    integer, parameter :: skip_step = 5000
    integer, parameter :: traning_step = 1000
    integer, parameter :: sample_num   = 5
    integer, parameter :: tr_data_size = traning_step*sample_num
    integer, parameter :: epoch        = 20
    integer, parameter :: ly_epoch     = 1000
    integer, parameter :: god_step = 0
    integer, parameter :: ly_skip_step = 0
    integer, parameter :: NOW = 2
    integer, parameter :: BEFORE = 1
    integer, parameter :: in_node = 1
    integer, parameter :: out_node = 1
    real(8), parameter :: g0 = 1.d0
    real(8), parameter :: gusai0 = 0.d0
    real(8), parameter :: epsi0 =1.d-6
    real(8) :: PI=3.14159265358979
    real(8) w_out(rc_node,out_node)
    real(8) w_in (in_node,rc_node)
    real(8) W_rnn(rc_node,rc_node)

  contains
    subroutine march(r,z,U_rc,S_rc,gusai,alpha,G)
        real(8) U_rc(:,:)
        real(8) S_rc(:,:)
        real(8) r(0:rc_step,1:rc_node)
        real(8) dr(0:rc_step,1:rc_node)
        real(8) z(0:rc_step,1:rc_node)
        real(8) u_tmp(1:in_node)
        real(8) r_tmp(1:rc_node)
        real(8) DF(rc_node, rc_node)
        real(8) wu(rc_node),wr(rc_node)
        real(8) gusai,alpha,G
        integer i,j,k,f,istep
!        write(*,*) dr_befor(1:par_node)
        do istep =1,rc_step
            wu=0.d0
            wr=0.d0
            do i=1,rc_node
            do j=1,in_node
                wu(i)=wu(i)+u_rc(istep,j)*W_in(j,i)
            enddo
            enddo
            do i=1,rc_node
            do j=1,rc_node
                wr(i)=wr(i)+r(istep-1,j)*W_rnn(j,i)
            enddo
            enddo
            do i=1,rc_node
                z(istep,i) =G*( wr(i)+wu(i)+gusai )
                r(istep,i)=tanh(z(istep,i))
            enddo
            do i = 1,out_node
	    	    s_rc(istep,i)  = 0.d0
	    	    do k = 1,rc_node
	    	        s_rc(istep,i) = s_rc(istep,i) + r(istep,k) * w_out(k,i)
	    	    enddo
	    	enddo
	    enddo
    end subroutine march
    
    subroutine create_dataset(dataset,trmax)
        real(8) dataset(:,:)
        real(8) X(3,1:3)
        integer istep,trmax
        X(BEFORE,:)=1.d0
!        open(10,file='./data/output_Runge_Lorenz.dat')
        write(*,*) "=========================================="
        write(*,*) "START   CREATE TRANING SET"
        write(*,*) "    TOTAL STEP =",trmax
        write(*,*) "=========================================="
        do istep = 1, skip_step
            call Runge_Kutta_method(X(NOW,1:3),X(BEFORE,1),X(BEFORE,2),X(BEFORE,3),istep)
!            if(mod(istep,10) == 0) write(10,*) istep ,x(NOW,1),x(NOW,2),x(NOW,3)
!            write(10,*) istep ,x(NOW,1),x(NOW,2),x(NOW,3)
            x(BEFORE,1:3) = x(NOW,1:3)
        enddo
!        close(10)
        do istep=1,trmax
            call Runge_Kutta_method(X(NOW,1:3),x(BEFORE,1),x(BEFORE,2),x(BEFORE,3),istep)
            x(BEFORE,1:3) = X(NOW,1:3)
            dataset(istep,1:3)= X(NOW,1:3)
        enddo
        call standard(dataset,3,trmax)
    end subroutine create_dataset
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	subroutine standard(a,data_node,step)
		real(8) a(:,:)
		integer data_node,step
		real(8) a_mean(data_node),a_var(data_node)
		integer i,j
		do i = 1,data_node
			a_mean(i)=mean(a(1:step,i),step)
		enddo
		do i = 1,data_node
			a_var(i)=variance(a(1:step,i),a_mean(i),step)
		enddo
		do j = 1,data_node
			do i=1,step
				a(i,j)=(a(i,j)-a_mean(j))/a_var(j)
			enddo
		enddo
	end subroutine standard
	function mean(a,step) result(out)
		real(8), intent(in) :: a(:)
		real(8) :: out
		integer i,step
		out=0.d0
		do i=1,step
			out = out + a(i)
		enddo
		out= out/dble(step)
	end function mean
	function variance(a,a_mean,step) result(out)
		real(8), intent(in) :: a(:),a_mean
		real(8) :: out
		integer i,step
		out=0.d0
		do i=1,step
			out = out + (a(i)-a_mean)**2
		enddo
		out= (out/dble(step))**0.5
	end function variance
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
!-----------------------------------------------------------------------------
!■オイラー法での近似
!-----------------------------------------------------------------------------
!    subroutine Euler_method(r_now,x,y,z,step)
!        real(8) r_now(1,1:3)
!        real(8) x,y,z
!        integer step
!
!        r_now(1,1) = x + (-par_a*(x - y) + y*z) * dt
!        r_now(1,2) = y + ((par_b - z)*x - y) *dt
!        r_now(1,3) = z + (x*y -par_c*z) *dt
!        open(20,file='output_001_Euler_x2y10z5.dat',position='append')
!        write(20,*) step ,r_now(1,1:3)
!        close(20)
!    end subroutine Euler_method
!======================================================
!-----------------------------------------------------------------------------
!■ルンゲクッタ法での近似
!-----------------------------------------------------------------------------
	subroutine Runge_Kutta_method(r_now,x,y,z,step)
		real(8) x,y,z
		real(8) r_now(1,1:3)
		real(8) k1(1:3),k2(1:3),k3(1:3),k4(1:3)
		real(8) k(1:3)
		integer i,j,step
		
		
        do j=1, int(dt/dt_Runge)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		    k1(1)=f1(x,y,z)
		    k1(2)=f2(x,y,z)
		    k1(3)=f3(x,y,z)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		    k2(1)=f1(x+k1(1)*0.5d0 ,y+k1(2)*0.5d0 , z+k1(3)*0.5d0)
		    k2(2)=f2(x+k1(1)*0.5d0, y+k1(2)*0.5d0 , z+k1(3)*0.5d0)
		    k2(3)=f3(x+k1(1)*0.5d0, y+k1(2)*0.5d0 , z+k1(3)*0.5d0)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		    k3(1)=f1(x+k2(1)*0.5d0, y+k2(2)*0.5d0 , z+k2(3)*0.5d0)
		    k3(2)=f2(x+k2(1)*0.5d0, y+k2(2)*0.5d0 , z+k2(3)*0.5d0)
		    k3(3)=f3(x+k2(1)*0.5d0, y+k2(2)*0.5d0 , z+k2(3)*0.5d0)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		    k4(1)=f1(x+k3(1), y+k3(2) , z+k3(3))
		    k4(2)=f2(x+k3(1), y+k3(2) , z+k3(3))
		    k4(3)=f3(x+k3(1), y+k3(2) , z+k3(3))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

		    do i=1,3
		    	k(i)=( k1(i) + 2.d0*k2(i) + 2.d0*k3(i) + k4(i) )/6.d0
		    enddo
		    x = x+k(1)
		    y = y+k(2)
		    z = z+k(3)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        enddo
		r_now(1,1) = x
		r_now(1,2) = y
		r_now(1,3) = z
!        open(20,file='output_001_Euler_x2y10z5.dat',position='append')
!        write(20,*) step ,r_now(1,1:3)
!        close(20)
    contains
        function f1(x,y,z)
	    	real(8) x,y,z
	    	real(8) f1
	    	f1=dt_Runge*(-par_a*(x-y))
	    end function f1
	    function f2(x,y,z)
	    	real(8) x,y,z
	    	real(8) f2
	    	f2=dt_Runge*((par_b-z)*x - y)
	    end function f2
	    function f3(x,y,z)
	    	real(8) x,y,z
	    	real(8) f3
	    	f3=dt_Runge*(x*y -par_c*z)
	    end function f3
	end subroutine Runge_Kutta_method
!======================================================
!======================================================

    subroutine Initialization_file
!        open(20,file='./data_out/lyapnov.dat',status='replace')
!        open(21,file="./data_out/tmp.dat",status='replace')
!        open(60,file="tmp2.dat",status='replace')
!        open(10,file='./data/output_Runge_Lorenz.dat',status='replace')
!        open(22,file="./data_out/rc_out.dat",status='replace')
!        close(10)
!        close(20)
!        close(21)
!        close(22)
!        close(60)
    end subroutine Initialization_file
    subroutine Ly_Calculation(abs_dr,lyapnov,step)
        real(8),intent(inout) :: lyapnov
        real(8),intent(in   ) ::  abs_dr
        integer step

        lyapnov =(lyapnov*dble(step-1) + log(abs_dr) /tau)/dble(step)
100 format(i5,a,f15.10,f15.10)
    end subroutine Ly_Calculation
    function sigmoid(x)
        real(8) x, sigmoid   ! 順番は関係ありません
        sigmoid = 1.d0 / (1.d0 + exp(-x) )
    end function sigmoid
    
    function d_sigmoid(x)
        real(8) x, d_sigmoid   ! 順番は関係ありません
        d_sigmoid = sigmoid(x)*(1-sigmoid(x))
    end function d_sigmoid
    
    function d_tanh(x)
        real(8) x, d_tanh   ! 順番は関係ありません
        d_tanh = 4.d0 /(( exp(x)+exp(-x) )**2)
    end function d_tanh
    subroutine calu_absdr(abs_dr,dr,ftau)
        real(8) abs_dr
        real(8),intent(in):: dr(0:tau,1:rc_node)
        integer fi,ftau
        abs_dr=0.d0
        do fi=1,rc_node
!            write(*,*) fi,dr(ftau,fi)
            abs_dr = abs_dr+dr(ftau,fi)**2
        enddo
        abs_dr=abs_dr**0.5d0
    end subroutine calu_absdr
    subroutine calu_ERR(S_rc_out,S_rc_data,ERR)
        real(8) S_rc_data(1:rc_step,1:out_node)
        real(8) S_rc_out(1:rc_step,1:out_node)
        real(8) ERR
        integer fi,fj,fk
        ERR=0.d0
        do fi=1,rc_step
            do fj=1,out_node
                ERR=ERR+(S_rc_data(fi,fj)-S_rc_out(fi,fj))**2
            enddo
        enddo
        ERR=(ERR/dble(rc_step) )**0.5d0
    end subroutine calu_ERR
    function rand_normal(mu,sigma)
    	real(8) mu,sigma
    	real(8) rand_normal
    	real(8) z,p1,p2
    	call random_number(p1)
    	call random_number(p2)
    	!write(*,*) p1,p2
        z=sqrt( -2.0*log(p1) ) * sin( 2.0*PI*p2 );
        rand_normal= mu + sigma*z
    end function rand_normal
end module lyapnov

program main
use lyapnov
implicit none
    real(8) U_tr(tr_data_size,in_node) !今は一次元、列サイズはトレーニング時間
    real(8) S_tr(tr_data_size,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8) U_rc(rc_step,in_node) !今は一次元、列サイズはトレーニング時間
    real(8) S_rc(rc_step,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8) S_rc_out(rc_step,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8) dr0(rc_node), lyap,abs_dr,abs_dr2
    real(8) r_before(1:rc_node)
    real(8) DF(rc_node, rc_node)
    real(8) dataset(1:rc_step+tr_data_size+god_step+100,par_node)
    real(8) r(0:rc_step,1:rc_node)
    real(8) dr(0:rc_step,1:rc_node)
    real(8) z(0:rc_step,1:rc_node)
    real(8) G_tmp,av_degree,p,ERR,ERR_ave,iepsi,err_tr
    integer step,epsi_adjust,itraning_step,isample_num
    integer istep,itau,iG,ily_epoch,iepoch,j,k,i,n
    character(6) :: cist,cist2
    
    
!    open(41,file="./data_out/lyapnov_end.dat",status='replace')

    open(60,file="./data_out/tmp2.dat",status='replace')
    close(60)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!■create_dataset
!-------------------------------------------------
    call create_dataset(dataset,rc_step+tr_data_size+god_step+100)
    U_tr(1:tr_data_size,1 )=dataset(1:tr_data_size,1)
    s_tr(1:tr_data_size,1)=dataset(1+god_step:tr_data_size+god_step,2 )
    U_rc(1:rc_step,1 )=dataset(1+tr_data_size:tr_data_size+rc_step,1)
    S_rc(1:rc_step,1)=dataset(1+tr_data_size+god_step:tr_data_size+rc_step+god_step,2)
    open(10,file='./data/output_Runge_Lorenz.csv')
    do i=1,tr_data_size+traning_step
        write(10,*) dataset(i,1),",",dataset(i+GOD_STEP,2),",",dataset(i,3)
    enddo
!    do i=1,tr_data_size
!        write(10,*) dataset(i,1),",",dataset(i+10,1),",",dataset(i+100,1)
!    enddo
    close(10)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	do itraning_step=traning_step,traning_step,200
!    isample_num=sample_num
    write(cist2,'(i4.4)') itraning_step
    open(41,file="./data_out/lyapnov_end_trstep."//cist2,status='replace')
    
    isample_num=int(dble(tr_data_size)/dble(itraning_step) )
    iepoch=int(dble(epoch)/dble(isample_num))
!    iepoch=epoch
    av_degree=1.d0
    do i=1,rc_node
    do k=1,rc_node
        w_rnn(i,k)=rand_normal(0.d0, 1.d0/(rc_node**0.5d0))
    enddo
    enddo
    do i=1,in_node
    do k=1,rc_node
        w_in(i,k)=rand_normal(0.d0, 1.d0/(rc_node**0.5d0))
    enddo
    enddo
    do i=1,rc_node
    do k=1,out_node
        w_out(i,k)=rand_normal(0.d0, 1.d0/(rc_node**0.5d0))
    enddo
    enddo
!    write(*,*) a(1:rc_node,1:rc_node)
!	call random_number(w_in)
!	call random_number(w_out)
!	w_in(:,:)=(2.d0*w_in(:,:)-1.d0)
!	w_out(:,:)=1.d-2 *(2.d0*w_out(:,:)-1.d0)
	iepsi=epsi0
	

    do ily_epoch=1,ly_epoch
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!■初期化
!-------------------------------------------------
    dr=0.d0
    lyap=0.d0
    r=1.d0
    abs_dr=0.d0
    ERR_ave=0.d0
    call Initialization_file
    call random_number(dr(0,:))
    write(cist,'(i4.4)') ily_epoch
    open(51,file='./data_renban2/rc_out.'//cist,status='replace')
    !write(*,*) dr(0,:)
!    call random_seed
!    call random_number(dr(0,:))
!    write(*,*) dr(0,:)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!■Wout calu
!-------------------------------------------------
!    call calu_wout(U_tr,S_tr)
!    call rc_traning_own_fortran(in_node,out_node,rc_node,traning_step,tau,gusai0,1.d0,G0,&
!                    u_tr,s_tr,U_rc,w_in,w_rnn,w_out)
    
    call rnn_traning_own_fortran(in_node,out_node,rc_node,itraning_step,rc_step,&
                    isample_num,epoch,iepsi,g0,gusai0,&
                    u_tr,s_tr,u_rc,w_out,w_rnn,w_in,err_tr)

!    write(*,*) "=========================================="
!    write(*,*) "START   CALUCULATE LY"
!    write(*,*) "=========================================="
!    dr(0,:) =1.d0
!    write(*,*) dr(0,:)
!    call calu_absdr(abs_dr,dr,0)
    r(0,:) =0.d0
    z(0,:) =0.d0
    call calu_absdr(abs_dr,dr,0)
    dr(0,:) = dr(0,:)/ abs_dr

!        write(*,*) dr(0,:)
!        write(*,*) dr(1:3)
    call march(r,z,U_rc,S_rc_out,gusai0,1.d0,G0)
    call calu_ERR(S_rc_out(1:rc_step,1:out_node),S_rc(1:rc_step,1:out_node),ERR)

    if(mod(ily_epoch,5)==0) then
        if(mod(istep,nint(lyapnov_step*1.0d0))==0) write(*,*) "---------------------------------------------------------"
        if(mod(istep,nint(lyapnov_step*1.0d0))==0) write(*,*)  ily_epoch
        if(mod(istep,nint(lyapnov_step*1.0d0))==0) write(*,*)  istep ,int(istep*100/rc_step),'%'
        if(mod(istep,nint(lyapnov_step*1.0d0))==0) write(*,*)  "ERR              ====", ERR
        if(mod(istep,nint(lyapnov_step*1.0d0))==0) write(*,*)  "err_tr           ====", err_tr
        if(mod(istep,nint(lyapnov_step*1.0d0))==0) write(*,*)  "epsi             ====", iepsi
        if(mod(istep,nint(lyapnov_step*1.0d0))==0) write(*,*)  "isample          ====", isample_num
        if(mod(istep,nint(lyapnov_step*1.0d0))==0) write(*,*)  "itraning_step    ====", itraning_step
    endif
    do i=1,rc_step
        write(51,*) S_rc_out(i,1),",", S_rc(i,1),",",U_rc(i,1)
    enddo
    close(51)
    write(41,*) ily_epoch,",",lyap,",",err_tr,",",ERR
    
    enddo
    close(41)
    
    enddo
200 format(a,f15.10)
end program
subroutine rnn_traning_own_fortran(in_node,out_node,rnn_node,traning_step,rnn_step,&
                    sample_num,epoch,epsi,g,gusai,&
                    u_tr_data0,s_tr_data0,u_rnn,w_outT,w_rnnT,w_inT,err_tr)
    implicit none
    integer(4), intent(inout) :: in_node,out_node,rnn_node,traning_step,rnn_step,sample_num,epoch
    real(8),    intent(inout) :: epsi,g,gusai
    real(8),    intent(inout) :: w_outT(rnn_node,out_node)
    real(8),    intent(inout) :: w_rnnT(rnn_node,rnn_node)
    real(8),    intent(inout) :: w_inT(in_node,rnn_node)
    
    real(8),    intent(inout) :: u_tr_data0(traning_step*sample_num,in_node) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) :: s_tr_data0(traning_step*sample_num,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8),    intent(inout) :: u_rnn(rnn_step,in_node) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) :: err_tr
    real(8)     Tre_CH(3,epoch)
    real(8)     u_tr_data(traning_step,in_node,sample_num) !今は一次元、列サイズはトレーニング時間
    real(8)     s_tr_data(traning_step,out_node,sample_num)  !出力次元数、列サイズはトレーニング時間
    
    real(8)     s_tr(traning_step,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8)     s_rnn(rnn_step,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8)     W_out(out_node,rnn_node)
    real(8)     W_rnn(rnn_node,rnn_node)
    real(8)     W_in(rnn_node,in_node)
    real(8)     out_dEdw(out_node,rnn_node)
    real(8)     rnn_dEdw(rnn_node,rnn_node)
    real(8)     in_dEdw(rnn_node,in_node)
    
    real(8)     out_delta(traning_step+1,out_node)
    real(8)     rnn_delta(traning_step+1,rnn_node)
    
    real(8)     R_tr(0:traning_step+1,rnn_node)
    real(8)     R_rnn(0:rnn_step+1,rnn_node)
    real(8)     z_tr(traning_step,rnn_node)
    real(8)     z_rnn(rnn_step,rnn_node)
    real(8)     drdz,tmp,u_tmp(1:in_node),r_tmp(rnn_node)
    real(8)     beta1,beta2,ipsi,alpha
    real(8)     Mt_out(out_node,rnn_node)
    real(8)     Mt_rnn(rnn_node,rnn_node)
    real(8)     Mt_in(rnn_node,in_node)
    real(8)     Vt_out(out_node,rnn_node)
    real(8)     Vt_rnn(rnn_node,rnn_node)
    real(8)     Vt_in(rnn_node,in_node)
    integer(4)  i,j ,k,iepo,isample,istep,total_step
!    write(*,*) size(u_tr_data0,1)
!    write(*,*) size(u_tr_data0,2)
    call create_test_to_sample(u_tr_data0,u_tr_data,traning_step,in_node )
    call create_test_to_sample(s_tr_data0,s_tr_data,traning_step,out_node)
    call inverse_matrix(W_inT,W_in,in_node,rnn_node)
    call inverse_matrix(W_rnnT,W_rnn,rnn_node,rnn_node)
    call inverse_matrix(W_outT,W_out,rnn_node,out_node)
!    call random_number(W_out)
!    call random_number(W_rnn)
!    call random_number(W_in )
!    W_out=W_out/(rnn_node)**0.5
 !   W_rnn=W_rnn/(rnn_node)**0.5
 !   W_in=W_in/(rnn_node)**0.5
    R_tr =0.d0
    R_rnn=0.d0
    s_tr =0.d0
    s_rnn=0.d0
    
    alpha=epsi
    beta1=0.9d0
    beta2=0.999d0
    ipsi =1.d-8
    Mt_out=0.d0
    Mt_rnn=0.d0
    Mt_in =0.d0
    Vt_out=0.d0
    Vt_rnn=0.d0
    Vt_in =0.d0
    err_tr=0.d0
    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    !トレーニング
    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    do iepo=1,epoch
        err_tr=0.d0
!        if(mod(iepo,10)==0) write(*,*) "epoch === " ,iepo
        do isample=1,sample_num
            !初期化
            out_dEdw=0.d0
            rnn_dEdw=0.d0
            in_dEdw=0.d0
            out_delta=0.d0
            rnn_delta=0.d0
            !順伝播計算
            total_step=(iepo-1)*sample_num +isample

            call rnn_forward3(U_tr_data,S_tr,R_tr,Z_tr,traning_step,rnn_node,isample)
            if(iepo==epoch) then
                do j=1,traning_step
                do i=1,out_node
                    err_tr=err_tr+(s_tr(j,i) - s_tr_data(j,i,isample))**2
                enddo
                enddo
                
            endif
            
            !逆伝播計算
            do istep=traning_step,1,-1
                
                !W_out
                out_delta(istep,:)= (s_tr(istep,:) - s_tr_data(istep,:,isample))
                do i=1,out_node
                do j=1,rnn_node
                    out_dEdw(i,j) = out_dEdw(i,j) + out_delta(istep,i) * R_tr(istep,j)
                enddo
                enddo

                !RNN
                do i=1,rnn_node
                    tmp=0.d0
                    drdz=0.d0
                    do k=1,out_node
                        tmp =tmp +(out_delta(istep,k) * W_out(k,i))
                    enddo
                    do k=1,rnn_node
                        tmp =tmp +(rnn_delta(istep+1,k) * W_rnn(k,i))
                    enddo
!                    drdz= 4.d0 /(( exp(z_tr(istep,i))+exp(-z_tr(istep,i)) )**2)
                    drdz=d_tanh(z_tr(istep,i))
                    rnn_delta(istep,i)=tmp*drdz*g
                enddo
                do i=1,rnn_node
                do j=1,rnn_node
                    rnn_dEdw(i,j) =rnn_dEdw(i,j)+ rnn_delta(istep,i) * R_tr(istep-1,j)
                enddo
                enddo

                !W_in
                do i=1,rnn_node
                do j=1,in_node
                    in_dEdw(i,j)=in_dEdw(i,j)+rnn_delta(istep,i) * u_tr_data(istep,j,isample)
                enddo
                enddo

            enddo
            !更新
            call SDM(in_dEdw ,epsi)
            call SDM(rnn_dEdw,epsi)
            call SDM(out_dEdw,epsi)


!            call ADAM( in_dEdw,Mt_in ,Vt_in ,alpha,beta1,beta2,epsi,total_step)
!            call ADAM(rnn_dEdw,Mt_rnn,Vt_rnn,alpha,beta1,beta2,epsi,total_step)
!            call ADAM(out_dEdw,Mt_out,Vt_out,alpha,beta1,beta2,epsi,total_step)
!
            do i=1,in_node
            do j=1,rnn_node
                W_in(j,i)  = W_in(j,i) - in_dEdw(j,i)
            enddo
            enddo
            do i=1,rnn_node
            do j=1,rnn_node
                W_rnn(j,i) =W_rnn(j,i) - rnn_dEdw(j,i)
            enddo
            enddo
!            write(*,*) rnn_dEdw(5,1),W_rnn(5,1)
!            write(*,*) R_tr(50,:)
            do i=1,rnn_node
            do j=1,out_node
!                W_out(j,i) =W_out(j,i) - epsi*(out_dEdw(j,i)+0.001d0*W_out(j,i) )
                W_out(j,i) =W_out(j,i) - out_dEdw(j,i)
            enddo
            enddo
            
        enddo
        R_tr(0,:)=R_tr(traning_step,:)
!        Tre_CH(1,iepo) = (sum(abs(out_dEdw)) )**2
!        Tre_CH(2,iepo) = (sum(abs(rnn_dEdw)) )**2
!        Tre_CH(3,iepo) = (sum(abs(in_dEdw)) )**2
!        write(*,*) out_dEdw(1,5),W_out(1,5)
    enddo
    err_tr=(err_tr/dble(sample_num*traning_step))**0.5d0
!    Tre_CH(1,:) =Tre_CH(1,:)/Tre_CH(1,1)
!    Tre_CH(2,:) =Tre_CH(2,:)/Tre_CH(2,1)
!    Tre_CH(3,:) =Tre_CH(3,:)/Tre_CH(3,1)

    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    !テスト
    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    R_rnn(0,:)=R_tr(traning_step,:)
    open(60,file="./data_out/tmp2.dat",position='append')
!    do i=1,sample_num/4
    do j=1,epoch
!        write(60,*) i,S_tr(i,1),S_tr_data(i,1,sample_num),u_tr_data(i,1,sample_num)
        !write(60,*) S_tr_data(j,1,i),u_tr_data(j,1,i),S_tr(j,1)
        write(60,*) Tre_CH(1:3,j)
    enddo
!    enddo
    close(60)
            

    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    !パイソンに出力
    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    call inverse_matrix(W_out,W_outT,out_node,rnn_node)
    call inverse_matrix(W_rnn,W_rnnT,rnn_node,rnn_node)
    call inverse_matrix(W_in ,W_inT ,rnn_node,in_node)
    
!    write(*,*) "+++++++++++++++++++++++++++++++"
!    write(*,*) "==============================="
!    write(*,*) "    EXIT     Fortran90 !    "
!    write(*,*) "-------------------------------"
!    write(*,*) "in_node     ",in_node
!    write(*,*) "out_node    ",out_node
!    write(*,*) "rnn_node     ",rnn_node
!    write(*,*) "traning_step",traning_step
!    write(*,*) "rnn_step     ",rnn_step
!    write(*,*) "-------------------------------"
!    write(*,*) "==============================="
!    write(*,*) "+++++++++++++++++++++++++++++++"
!    write(*,*) ""
contains
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
     subroutine ADAM(dEdw,Mt,Vt,alpha,beta1,beta2,ipsi,fstep)
        real(8) dEdw(:,:),Mt(:,:),Vt(:,:)
        real(8) alpha,beta1,beta2,ipsi
        integer size1,size2
        integer fstep,fi,fj
        
        size1= size(dEdw,1)
        size2= size(dEdw,2)
        
        do fi=1,size2
        do fj=1,size1
            Mt(fj,fi)   = beta1*Mt(fj,fi) +(1-beta1) * dEdw(fj,fi)
            Vt(fj,fi)   = beta2*Vt(fj,fi) +(1-beta2) * dEdw(fj,fi)**2
            dEdw(fj,fi) = alpha * (Mt(fj,fi)/(1.d0-beta1**fstep) )/((Vt(fj,fi)/(1.d0-beta2**fstep) )**0.5d0 + ipsi)
        enddo
        enddo
        
     end subroutine ADAM
     subroutine SDM(dEdw,ipsi)
        real(8) dEdw(:,:)
        real(8) ipsi
        integer size1,size2
        integer fi,fj

        size1= size(dEdw,1)
        size2= size(dEdw,2)
        
        do fi=1,size2
        do fj=1,size1
            dEdw(fj,fi)=ipsi* dEdw(fj,fi)
        enddo
        enddo
     end subroutine SDM
     subroutine rnn_forward3(u_f,s_f,R_f,z_f,nstep,nnode,nsample)
        real(8) u_f(:,:,:),s_f(:,:),R_f(:,:),z_f(:,:)
        integer nstep,nnode
        real(8) wu(nnode), wr(nnode)
        integer f1,f2,f,fistep,nsample
!        call rnn_function(R_f,u_f,z_f,nstep)
!!!!!!!!

        do fistep=1,nstep
            wu(:)=0.d0
            wr(:)=0.d0
            
            do f1=1,rnn_node
                do f2=1,in_node
                    wu(f1) = wu(f1) + W_in(f1,f2) * u_f(fistep,f2,nsample)
                enddo
                do f2=1,rnn_node
                    wr(f1) = wr(f1) + W_rnn(f1,f2) *R_f(fistep-1,f2)
                enddo
                z_f(fistep,f1)=g*(wu(f1) + wr(f1)+gusai)
                R_f(fistep,f1) = tanh(z_f(fistep,f1))
            enddo
            do f1=1,out_node
                s_f(fistep,f1)=0.d0
                do f2=1,rnn_node
                    s_f(fistep,f1) = s_f(fistep,f1)+ W_out(f1,f2)*R_f(fistep,f2)
                enddo
            enddo
!        write(*,*) s_f(fistep,:)
        enddo
        

!!!!!!!!!!
    end subroutine rnn_forward3

    subroutine rnn_forward(u_f,s_f,R_f,z_f,nstep,nnode)
        real(8) u_f(:,:),s_f(:,:),R_f(:,:),z_f(:,:)
        integer nstep,nnode
        real(8) wu(nnode), wr(nnode)
        integer f1,f2,f,fistep
!        call rnn_function(R_f,u_f,z_f,nstep)
!!!!!!!!

        do fistep=1,nstep
            wu(:)=0.d0
            wr(:)=0.d0
            
            do f1=1,rnn_node
                do f2=1,in_node
                    wu(f1) = wu(f1) + W_in(f1,f2) * u_f(fistep,f2)
                enddo
                do f2=1,rnn_node
                    wr(f1) = wr(f1) + W_rnn(f1,f2) *R_f(fistep-1,f2)
                enddo
                z_f(fistep,f1)=g*(wu(f1) + wr(f1))

                R_f(fistep,f1) = tanh(z_f(fistep,f1) )
            enddo
            do f1=1,out_node
                s_f(fistep,f1)=0.d0
                do f2=1,rnn_node
                    s_f(fistep,f1) = s_f(fistep,f1)+ W_out(f1,f2)*R_f(fistep,f2)
                enddo
            enddo
!        write(*,*) s_f(fistep,:)
        enddo
        

!!!!!!!!!!
    end subroutine rnn_forward

    subroutine create_test_to_sample(A_A,B_B,fstep,fnode)
        real(8) A_A(:,:),B_B(:,:,:)
        integer fnode,fstep
        integer f1,f2,f3
        
        do f3=1,sample_num
        do f2=1,fnode
        do f1=1,fstep
            B_B(f1,f2,f3) = A_A(f1+(f3-1)*fstep,f2 )
        enddo
        enddo
        enddo
    end subroutine create_test_to_sample
    subroutine inverse_matrix(A_A,B_B,v2,v1)
        real(8) A_A(:,:),B_B(:,:)
        integer v1,v2,v11,v22,v33
        integer f1,f2,f3
        
        v11=size(A_A,1)
        v22=size(A_A,2)


        do f1=1,v1
        do f2=1,v2
            B_B(f1,f2) = A_A(f2,f1)
        enddo
        enddo

    end subroutine inverse_matrix
end subroutine rnn_traning_own_fortran

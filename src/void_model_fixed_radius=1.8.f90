! ==========================================
! 1. 定义精度模块
! ==========================================
module mod_precision
    implicit none
    integer, parameter :: dp = selected_real_kind(15, 307)
end module mod_precision

! ==========================================
! 2. 定义随机数模块
! ==========================================
module mod_random
    use mod_precision
    implicit none
    private
    public :: init_rng, get_uniform

contains
    !> 初始化随机数种子 (固定种子以便对比)
    subroutine init_rng(seed_in)
        integer, intent(in), optional :: seed_in
        integer :: n, i, seed_out
        integer, allocatable :: seed_arr(:)
        call random_seed(size = n)
        allocate(seed_arr(n))
        if (present(seed_in)) then
            seed_arr = seed_in
        else
            seed_arr = 123456789
        end if
        call random_seed(put = seed_arr)
        deallocate(seed_arr)
    end subroutine init_rng

    !> 获取 [0, 1] 之间的均匀分布随机数
    function get_uniform() result(val)
        real(dp) :: val
        call random_number(val)
    end function get_uniform
end module mod_random

! ==========================================
! 3. 主程序：批量生成数据
! ==========================================
program void_model_batch_generator
    use mod_precision
    use mod_random
    implicit none

    ! --- 变量声明区 ---
    character(len=100) :: filename
    real(dp) :: L, r0, total_area, target_void_area, current_void_area, radius, eta_target
    real(dp) :: rho, rho_min, rho_max
    integer :: i, j, seed_val, N_circles, unit_out, max_iterations, n_rho
    real(dp), allocatable :: x_centers(:), y_centers(:), radii(:)
    real(dp), allocatable :: x_temp(:), y_temp(:), r_temp(:)
    real(dp), parameter :: PI = 3.141592653589793_dp

    ! --- 1. 参数初始化 (对齐文献的物理场景) ---
    L = 500.0_dp      ! 将系统尺寸调整为 500m，对齐文献比例尺
    r0 = 1.8_dp       ! 设定固定的典型圆半径 1.8m
    total_area = L * L  ! 计算系统总面积
    rho_min = 0.20_dp
    rho_max = 0.32_dp
    n_rho = 10
   
    max_iterations = 4000000  ! 提高上限，以防高密度下需要更多圆

    ! --- 2. 参数扫描循环 ---
    print *, "Arctic Ponds Percolation Simulation: Batch Mode"
    print *, "(使用固定半径 r0=1.8m, 系统边长 L=500m)"
    print *, "---------------------------------------------"

    do j = 1, n_rho
        seed_val = 1011
        call init_rng(seed_val)
        rho = rho_min + (j-1) * (rho_max - rho_min) / (n_rho - 1)
        write(filename, "(A, F5.3, A)") "voids_rho_", rho, ".txt"
        print *, "Generating: ", trim(filename)

        ! 利用 Poisson Boolean 模型计算目标面积
        eta_target = -log(rho)
        target_void_area = eta_target * total_area
        current_void_area = 0.0_dp
        N_circles = 0
        
        allocate(x_centers(10000), y_centers(10000), radii(10000))

        ! 动态生成圆形
        do i = 1, max_iterations
            N_circles = N_circles + 1
            
            ! 内存动态扩展
            if (N_circles > size(x_centers)) then
                allocate(x_temp(N_circles-1), y_temp(N_circles-1), r_temp(N_circles-1))
                x_temp = x_centers(1:N_circles-1)
                y_temp = y_centers(1:N_circles-1)
                r_temp = radii(1:N_circles-1)
                deallocate(x_centers, y_centers, radii)
                allocate(x_centers(N_circles*2), y_centers(N_circles*2), radii(N_circles*2))
                x_centers(1:N_circles-1) = x_temp
                y_centers(1:N_circles-1) = y_temp
                radii(1:N_circles-1) = r_temp
                deallocate(x_temp, y_temp, r_temp)
            end if
            
            x_centers(N_circles) = get_uniform() * L
            y_centers(N_circles) = get_uniform() * L
            radius = -r0 * log(1.0_dp - get_uniform())  ! 改为指数分布以模拟实际情况
            radii(N_circles) = radius
            
            current_void_area = current_void_area + PI * radius**2
            
            if (current_void_area >= target_void_area) exit
        end do

        ! 数据输出
        open(newunit=unit_out, file=trim(filename), status='replace')
        write(unit_out, '(A)') "# Arctic Ponds Geometry Data"
        write(unit_out, '(A, F10.5)') "# Void Area Ratio (rho): ", rho
        write(unit_out, '(A, I8)')    "# Number of circles (N): ", N_circles
        write(unit_out, '(A, F15.6)') "# Total void area (generated): ", current_void_area
        write(unit_out, '(A, F15.6)') "# Total area (L²): ", total_area
        write(unit_out, '(A)') "# Columns: ID, x_center, y_center, radius"
        
        do i = 1, N_circles
            write(unit_out, '(I8, 3(1X, F12.6))') i, x_centers(i), y_centers(i), radii(i)
        end do
        close(unit_out)
        
        print '(A, I8, A, F10.6)', "  Generated ", N_circles, " circles, actual area ratio: ", current_void_area/total_area

        deallocate(x_centers, y_centers, radii)
    end do

    print *, "---------------------------------------------"
    print *, "All", n_rho, "data files generated successfully."
end program void_model_batch_generator
디렉토리별 파일 특징

*모든 디렉토리에는 숫자로 된 데이터와 이를 plot한 그림이 존재합니다
*특정 파라미터의 값(delta, t, t_SO, lda, B, n, m)을 갖는 데이터의 경우 파라미터를 파일명으로 지정했습니다.
*파라미터 값을 바꿔가며(ex. size n =15~32) 저장한 데이터의 파일명은 가로축 데이터의 정보가 괄호안에 들어가 있습니다.

1D_density: 다이아몬드의 왼쪽 위 edge를 따라가는 density
1D_density_fit: 위의 데이터에 대해 지수 감쇠 함수를 피팅했을 때의 그 피팅된 함수

2D_density: 좌표(x,y)에 대한 가중치; density가 약한 곳은 배경과 구분이 잘 안 될 수 있기 때문에 plot할 때 어느정도 offset 설정이 필요할 수 있습니다

Bandgap_size: 가로축 size(n), 세로축 Band gap(E/t); 따로 역수를 취하진 않았습니다

Corner_states eigenvalue&eigenvector: corner state 2개에 대한 고유값과 고유벡터

Eigenvalue: 기본 시스템(t_SO=0.1, lda=0.2, n=30, m=30)에 대한 state 번호와 고유값

Ribbon_band_gap: 리본 구조(육각형 55개, t_SO=0.1, lda=0.2)의 edge gap

Splitting_size: size별로의 splitting
Splitting_size_fit: 위의 데이터를 y만 로그를 취한 후 직선으로 피팅한 뒤, 다시 지수함수로 변환한 값이 세로축이고, 가로축은 위의 데이터와 동일합니다

xi_lda: 람다(H_y)값을 변화시키며 구한 xi값
xi_lda_fit: 위의 데이터를 power함수에 피팅한 값입니다.

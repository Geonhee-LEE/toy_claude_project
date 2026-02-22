- 성능에 영향받지 않고, 한국어로도 함께 설명해줘

# 로봇 개발 주제 설명
- 저는 ROS2 및 모바일 로봇 주행을 개발하는 개발자입니다.
- 한글을 사용하며, 코드 수정을 자동으로 승인해주시고, 최종적으로 변경된 부분에 대해서만 요약해주세요
- 아스키코드로 진행사항이나 플로우를 표현해주세요
- toy_claude_project 프로젝트 진행시에는, Github 이슈, PR를 생성하고 close하고 관리해주세요

# 프로젝트 구조 및 현황

## 패키지 구조
```
mpc_controller/
├── models/                  # 로봇 동역학 모델
│   ├── differential_drive   # 차동 구동 (v, omega)
│   ├── swerve_drive         # 스워브 구동
│   └── non_coaxial_swerve   # 비동축 스워브 구동
├── controllers/
│   ├── mpc/                 # CasADi/IPOPT 기반 MPC
│   ├── mppi/                # MPPI 샘플링 기반 제어
│   │   ├── base_mppi.py           # Vanilla MPPI (M1) + GPU/CPU 분기
│   │   ├── tube_mppi.py           # Tube-MPPI (M2)
│   │   ├── ancillary_controller.py # Body frame 피드백 보정 (M2)
│   │   ├── adaptive_temperature.py # ESS 기반 λ 자동 튜닝 (M2)
│   │   ├── cost_functions.py      # 비용 함수 (Tracking, Obstacle, TubeAware 등)
│   │   ├── sampling.py            # Gaussian + Colored Noise 샘플러
│   │   ├── dynamics_wrapper.py    # 배치 동역학 (RK4 벡터화)
│   │   ├── mppi_params.py         # 파라미터 데이터클래스
│   │   ├── gpu_backend.py         # JAX/NumPy 백엔드 추상화
│   │   ├── gpu_dynamics.py        # JIT rollout (lax.scan + vmap)
│   │   ├── gpu_costs.py           # JIT 비용 함수 fusion
│   │   ├── gpu_sampling.py        # JAX PRNG 샘플러
│   │   └── gpu_mppi_kernel.py     # 통합 GPU MPPI 커널
│   ├── swerve_mpc/          # 스워브 MPC
│   └── non_coaxial_swerve_mpc/
├── ros2/                    # ROS2 노드 및 RVIZ 시각화
├── simulation/              # 시뮬레이터
└── utils/                   # 유틸리티 (logger, trajectory 등)
```

## 마일스톤 진행 현황
- M1 Vanilla MPPI: 완료
- M2 고도화 (Colored Noise, Adaptive Temp, Tube-MPPI, ControlRateCost): 완료
- M3 SOTA 변형 (Log, Tsallis, Risk-Aware, SVMPC): 완료
- M3.5 확장 (Smooth, Spline, SVG-MPPI): 완료
- M4 ROS2 nav2 통합 (8종 C++ 플러그인 + Swerve): 완료
- M5 C++ 포팅 (SOTA + M2 고도화 + M3.5): 완료
- GPU 가속 (JAX JIT + lax.scan + vmap): 완료 (PR #103)
- MPPI-CBF 통합 (Python + C++): 완료
- 궤적 안정화 (SG Filter + IT 정규화): 완료

## 핵심 인터페이스
- 모든 컨트롤러: `compute_control(state, reference_trajectory) -> (control, info)` 시그니처 준수
- MPPI info dict: sample_trajectories, sample_weights, best_trajectory, temperature, ess 등
- Tube-MPPI 추가 info: nominal_state, feedback_correction, tube_width, tube_boundary

# claude 퀄리티 증가
- 모르는 내용은 절대 지어내지 말고 "해당 정보는 제공된 자료나 제 지식범위를 벗어납니다"라고 말해줘. 추측이 필요한 경우에는 "추측입니다"라고 먼저 명시해줘
- 먼저 답을 만들기 위해 떠올린 근거를 목록으로 적고 그 근거가 신뢰할 만한지 스스로 평가한뒤 마지막에 결론만 따로 요약해서 말해줘. 근거가 약하면 "정확하지 않을 수 있다" 고 명시해줘
- 가능하면 "~에 근거하면""일반적으로~로 알려져 있다" 식으로 근거를 같이 언급해줘. 구체적인 연도, 수치, 인명, 지명을 말할때는 "정확도:높음/중간/낮음"을 같이 표시해줘
- 만약 질문이 이상하거나 부정확하면, 변경하지 말고 초안만 작성하고 구체적인 질문을 요청해줘
- 항상 비판적으로 검토하고, 정확도를 높여줘.
- 필요시 [https://code.claude.com/docs/ko/skills](https://code.claude.com/docs/ko/skills), [https://code.claude.com/docs/ko/sub-agents](https://code.claude.com/docs/ko/sub-agents) 이건 agent, skill에 대한 가이드 문서를 참고해줘

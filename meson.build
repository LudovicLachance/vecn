project('ludo_vecn',
    'cpp',
    version : '1.0.0',
    license : 'None',
)

ludo_vecn_dep = declare_dependency(
    include_directories: include_directories('include')
)

if not meson.is_subproject()
install_headers('include/ludo/vecn.hpp', subdir: 'ludo')

pkgc = import('pkgconfig')
pkgc.generate(name: 'ludo_vecn',
    version: meson.project_version(),
    description: 'vector and matrice manipulation'
)
endif
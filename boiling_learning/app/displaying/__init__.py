from typing import Final

units: Final = {
    'mse': '\\si{(\\watt\\per\\square\\cm)\\squared}',
    'rmse': '\\si{\\watt\\per\\square\\cm}',
    'rms': '\\si{\\watt\\per\\square\\cm}',
    'mae': '\\si{\\watt\\per\\square\\cm}',
    'mape': '\\si{\\percent}',
    'r2': '---',
    'power': '\\si{\\watt}',
    'heat flux': '\\si{\\watt\\per\\square\\cm}',
    'temperature': '\\si{\\celsius}',
}

glossary: Final = {
    'power': 'q',
    'heat flux': 'q^{\\prime\\prime}',
    'wall superheat': '\\Delta T_{\\mathrm{sat}}',
    'downscaling factor': 'f_\\text{ds}',
    'large wire ds': '\\mathcal{D}^{\\mathrm{LW}}',
    'small wire ds': '\\mathcal{D}^{\\mathrm{SW}}',
    'horizontal ribbon ds': '\\mathcal{D}^{\\mathrm{HR}}',
    'vertical ribbon ds': '\\mathcal{D}^{\\mathrm{VR}}',
}

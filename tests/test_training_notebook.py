"""
Tests for NEXUS notebook generator.
"""

from nexus.training.notebook_generator import NotebookGenerator


def test_colab_notebook_generation():
    """Test Colab notebook generation."""
    gen = NotebookGenerator()
    
    result = gen.generate_colab_notebook(
        job_id='test_colab_job',
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
        training_config={'method': 'qlora', 'config': {}},
        use_drive=False,
    )
    
    assert result['job_id'] == 'test_colab_job'
    assert result['provider'] == 'colab'
    assert 'notebook' in result
    assert 'output_path' in result
    assert 'instructions' in result
    
    # Check notebook structure
    nb = result['notebook']
    assert 'cells' in nb
    assert 'metadata' in nb
    assert 'nbformat' in nb
    assert nb['nbformat'] == 4
    
    # Check cells
    cells = nb['cells']
    assert len(cells) > 0
    
    # First cell should be markdown
    assert cells[0]['cell_type'] == 'markdown'
    
    # Check instructions
    assert len(result['instructions']) > 0
    assert any('GPU' in inst for inst in result['instructions'])


def test_colab_notebook_with_drive():
    """Test Colab notebook with Google Drive."""
    gen = NotebookGenerator()
    
    result = gen.generate_colab_notebook(
        job_id='test_colab_drive',
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
        training_config={'method': 'qlora', 'config': {}},
        use_drive=True,
    )
    
    # Check that drive mount cell exists
    cells = result['notebook']['cells']
    drive_cells = [c for c in cells if c['cell_type'] == 'code' and any('drive' in str(line).lower() for line in c.get('source', []))]
    assert len(drive_cells) > 0


def test_kaggle_notebook_generation():
    """Test Kaggle notebook generation."""
    gen = NotebookGenerator()
    
    result = gen.generate_kaggle_notebook(
        job_id='test_kaggle_job',
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
        training_config={'method': 'qlora', 'config': {}},
    )
    
    assert result['job_id'] == 'test_kaggle_job'
    assert result['provider'] == 'kaggle'
    assert 'notebook' in result
    assert 'instructions' in result
    
    # Check notebook structure
    nb = result['notebook']
    assert 'cells' in nb
    assert 'metadata' in nb
    
    # Check kaggle metadata
    assert 'kaggle' in nb['metadata']
    assert nb['metadata']['kaggle']['enable_gpu'] is True
    assert nb['metadata']['kaggle']['enable_internet'] is True


def test_notebook_cell_types():
    """Test notebook cell types."""
    gen = NotebookGenerator()
    
    result = gen.generate_colab_notebook(
        job_id='test_cell_types',
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
        training_config={'method': 'qlora', 'config': {}},
    )
    
    cells = result['notebook']['cells']
    cell_types = [c['cell_type'] for c in cells]
    
    # Should have both markdown and code cells
    assert 'markdown' in cell_types
    assert 'code' in cell_types


def test_notebook_training_code():
    """Test that notebook contains training code."""
    gen = NotebookGenerator()
    
    result = gen.generate_colab_notebook(
        job_id='test_training_code',
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        dataset_id='test/dataset',
        training_config={'method': 'qlora', 'config': {}},
    )
    
    # Collect all code cell sources
    code_sources = []
    for cell in result['notebook']['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            code_sources.append(source)
    
    # Should contain training-related code
    all_code = '\n'.join(code_sources)
    assert 'Trainer' in all_code or 'trainer' in all_code
    assert 'model' in all_code.lower()
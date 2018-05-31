#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <atomic>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/thread_pool.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"

namespace caffe {

class Solver;

/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
class Net {
 public:
  explicit Net(const NetParameter& param,
      size_t solver_rank = 0U,
      Flag* solver_init_flag = nullptr,
      const Net* root_net = nullptr,
      bool inner_net = false,
      int level = 0,
      const vector<string>* stages = NULL);
  Net(const string& param_file,
      Phase phase,
      size_t solver_rank = 0U,
      Flag* solver_init_flag = nullptr,
      const Net* root_net = nullptr,
      bool inner_net = false,
      int level = 0,
      const vector<string>* stages = NULL);
  ~Net();

  /// @brief Initialize a network with a NetParameter.
  void Init(const NetParameter& param);

  /**
   * @brief Run Forward with the input Blob%s already fed separately.
   *
   * You can get the input blobs using input_blobs().
   */
  const vector<Blob*>& Forward(float *loss = nullptr);

  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.
   */
  float ForwardFromTo(int start, int end);
  float ForwardFrom(int start);
  float ForwardTo(int end);
  /// @brief DEPRECATED; set input blobs then use Forward() instead.
  const vector<Blob*>& Forward(const vector<Blob*> & bottom, float *loss = nullptr);

  /**
   * @brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
   */
  void ClearParamDiffs();

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
  void Backward(bool apply_update = true);
  void BackwardFromToAu(int start, int end, bool apply_update);
  void BackwardFromTo(int start, int end);
  void BackwardFrom(int start);
  void BackwardTo(int end);
  void Finalize();

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  void Reshape();
  void ReduceAndUpdate(int type_id);

  float ForwardBackward(bool apply_update = true);

  /// @brief Updates the network weights based on the diff values computed.
  void Update();
  /**
   * @brief Shares weight data of owner blobs with shared blobs.
   *
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   */
  void ShareWeights();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   */
  void ShareTrainedLayersWith(const Net* other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string trained_filename);
  void CopyTrainedLayersFromHDF5(const string trained_filename);
  /// @brief Writes the net to a proto.
  void ToProto(NetParameter* param, bool write_diff = false, bool write_data = true) const;
  /// @brief Writes the net to an HDF5 file.
  void ToHDF5(const string& filename, bool write_diff = false) const;

  /// @brief returns the network name.
  const string& name() const { return name_; }
  /// @brief returns the layer names
  const vector<string>& layer_names() const { return layer_names_; }
  /// @brief returns the blob names
  const vector<string>& blob_names() const { return blob_names_; }
  /// @brief returns the blobs
  const vector<shared_ptr<Blob>>& blobs() const {
    return blobs_;
  }
  /// @brief returns the layers
  const vector<shared_ptr<LayerBase>>& layers() const {
    return layers_;
  }
  /// @brief returns the phase: TRAIN or TEST
  Phase phase() const { return phase_; }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  const vector<vector<Blob*> >& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  const vector<vector<Blob*> >& top_vecs() const {
    return top_vecs_;
  }
  /// @brief returns the ids of the top blobs of layer i
  const vector<int> & top_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
    return top_id_vecs_[i];
  }
  /// @brief returns the ids of the bottom blobs of layer i
  const vector<int> & bottom_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
    return bottom_id_vecs_[i];
  }
  const vector<vector<bool> >& bottom_need_backward() const {
    return bottom_need_backward_;
  }
  const vector<float>& blob_loss_weights() const {
    return blob_loss_weights_;
  }
  const vector<bool>& layer_need_backward() const {
    return layer_need_backward_;
  }
  /// @brief returns the parameters
  const vector<shared_ptr<Blob>>& params() const {
    return params_;
  }
  const vector<shared_ptr<Blob>>& learnable_params() const {
    return learnable_params_;
  }
  const vector<shared_ptr<Blob>>& learnable_params_mapped() const {
    return learnable_params_mapped_;
  }

  const vector<Type>& learnable_types(bool reset = false);

  /// @brief returns the learnable parameter learning rate multipliers
  const vector<float>& params_lr() const { return params_lr_; }
  const vector<bool>& has_params_lr() const { return has_params_lr_; }
  /// @brief returns the learnable parameter decay multipliers
  const vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  const vector<bool>& has_params_decay() const {
    return has_params_decay_;
  }
  const map<string, int>& param_names_index() const {
    return param_names_index_;
  }
  const vector<int>& param_owners() const { return param_owners_; }
  const vector<string>& param_display_names() const {
    return param_display_names_;
  }

  const pair<int, int>& param_layer_indices(int param_id) {
    return param_layer_indices_[param_id];
  }

  /// @brief Input and output blob numbers
  int num_inputs() const { return net_input_blobs_.size(); }
  int num_outputs() const { return net_output_blobs_.size(); }
  const vector<Blob*>& input_blobs() const {
    return net_input_blobs_;
  }
  const vector<Blob*>& output_blobs() const {
    return net_output_blobs_;
  }
  const vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  const vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  bool has_blob(const string& blob_name) const;
  const shared_ptr<Blob> blob_by_name(const string& blob_name) const;
  bool has_layer(const string& layer_name) const;
  const shared_ptr<LayerBase> layer_by_name(const string& layer_name) const;

  void set_debug_info(const bool value) { debug_info_ = value; }

  // Helpers for Init.
  /**
   * @brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
   */
  static void FilterNet(const NetParameter& param,
      NetParameter* param_filtered);
  /// @brief return whether NetState state meets NetStateRule rule
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
      const string& layer_name);

  /// @brief set a Solver and layers properties for this net
  void set_solver(Solver* s);

  unsigned int batch_per_solver() const {
    return batch_per_solver_;
  }

  Solver* parent_solver() {
    return solver_;
  };

  bool trained_layers_shared() const {
    return trained_layers_shared_;
  }

  void InitializeLearnableDiffSpace(int type_id);

  void wait_layers_init() {
    for (Flag* flag : layer_inititialized_flags_) {
      flag->wait();
    }
  }

  float global_grad_scale() const {
    return global_grad_scale_coeff_;
  }

  size_t infer_count() const {
    return infer_count_;
  }

  size_t solver_rank() const {
    return solver_rank_;
  }

  bool global_grad_scale_enabled() const {
    return has_global_grad_scale_param_ && global_grad_scale_param_ > 0.F;
  }

  bool inner_net() const {
    return inner_net_;
  }

  void update_grad_scale();

  std::string print_current_device() const {
    std::ostringstream os;
    os << (phase_ == TRAIN ? "[" : "(")
       << Caffe::current_device()
       << (phase_ == TRAIN ? "]" : ")");
    return os.str();
  }

  template <typename Ftype, typename Btype>
  size_t prefetch_bytes() {
    size_t bytes = 0UL;
    for (const shared_ptr<LayerBase>& layer : layers_) {
      if (typeid(*layer) == typeid(DataLayer<Ftype, Btype>) ||
          typeid(*layer) == typeid(AnnotatedDataLayer<Ftype, Btype>)) {
        bytes += reinterpret_cast<DataLayer<Ftype, Btype>*>(layer.get())->prefetch_bytes();
      }
    }
    return bytes;
  }

  //Quantization
  void StartQuantization();
  void FinishQuantization();
  void EnableQuantizationForSelectedLayers();

  void ClearQuantizationRangeInLayers();
  void UpdateQuantizationRangeInLayers();
  void CopyQuantizationRangeInLayers();
  void SetQuantizationParamsLayerWeights(const int layer_id);
  void SetQuantizationParamsLayerInput(const int layer_id);
  void SetQuantizationParamsLayerOutput(const int layer_id);
  void SetQuantizationParams();
  void DisplayQuantizationParams();
  void DisableQuantization();

  template <typename Dtype> void Convert2FixedPoint_cpu(Dtype* data, const int cnt,
      const int bw, int fl, bool unsigned_data, bool clip) const;

  int EstimateAbsBits(float val);
  void EstiamteQScaleParams(float min, float max, int bitwidth, bool power2_quant,
      bool unsigned_data, bool apply_offset, QuantizationParameter::QParams& qparam_xx);

  //Sparsity
  int GetSparsity(std::map<std::string, std::pair<int,int> >& sparsity_map);
  int GetConnectivitySparsity(std::map<std::string, std::pair<int,int> >& sparsity_map);
  void FindAndApplyThresholdNet(float threshold_fraction_low, float threshold_fraction_mid, float threshold_fraction_high,
      float threshold_value_maxratio, float threshold_value, float threshold_step_factor, bool verbose = true);
  void FindAndApplyChannelThresholdNet(float threshold_fraction_low, float threshold_fraction_mid, float threshold_fraction_high,
      float threshold_value_maxratio, float threshold_value, float threshold_step_factor, bool verbose = true);
  void ApplySparseModeConnectivity();  
  void StoreSparseModeConnectivity(SparseMode mode);
  float DisplaySparsity(bool verbose);
  float DisplayConnectivitySparsity(bool verbose);
  
  //Batch Norm  Optimization
  template <typename Dtype> void OptimizeNet();
  
  //Some utility functions
  vector<int> GetTopLayerIds(int layer_id, bool only_one=true);
  vector<int> GetBottomLayerIds(int layer_id, bool only_one=true);
  string GetTopLayerType(int layer_id, bool only_one=true);
  string GetBottomLayerType(int layer_id, bool only_one=true);
  string GetTopLayerType2(int layer_id, bool only_one=true);
  string GetBottomLayerType2(int layer_id, bool only_one=true);
  vector<const QuantizationParameter::QParams*> GetBottomLayerQParams(int layer_id);

 protected:
  // Helpers for Init.
  /// @brief Append a new top blob to the net.
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(const int param_id);
  /// @brief Multi-GPU reduction for a particular parameter.
  void Reduce(int type_id, int param_id);
  /// @brief Multi-GPU reduction for a particular bucket of parameters.
  void ReduceBucket(int type_id, size_t count, Type bucket_type, void* bucket);
  size_t received_contiguous_count(int type_id, const std::set<int>& au_ids, int& from);

  size_t lp_aligned_count(int id) const {
    return align_up<6>((size_t)learnable_params_[id]->count());
  }

  size_t lp_size(int id) const {
    return tsize(learnable_params_[id]->diff_type());
  }

  void add_wgrad_sq(float wgrad_sq);
  float wgrad_sq();

  /// @brief The network name
  string name_;
  /// @brief The phase: TRAIN or TEST
  Phase phase_;
  /// @brief Individual layers in the net
  vector<shared_ptr<LayerBase> > layers_;
  vector<string> layer_names_;
  map<string, int> layer_names_index_;
  vector<bool> layer_need_backward_;
  /// @brief the blobs storing intermediate results between the layer.
  vector<shared_ptr<Blob> > blobs_;
  vector<string> blob_names_;
  map<string, int> blob_names_index_;
  vector<bool> blob_need_backward_;
  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers.
  vector<vector<Blob*> > bottom_vecs_;
  vector<vector<int> > bottom_id_vecs_;
  vector<vector<bool> > bottom_need_backward_;
  /// top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob*> > top_vecs_;
  vector<vector<int> > top_id_vecs_;
  /// Vector of weight in the loss (or objective) function of each net blob,
  /// indexed by blob_id.
  vector<float> blob_loss_weights_;
  vector<vector<int> > param_id_vecs_;
  vector<int> param_owners_;
  vector<string> param_display_names_;
  vector<pair<int, int> > param_layer_indices_;
  /// (layer, blob) -> param_id map
  map<pair<int, int>, int> layer_index_params_;
  map<string, int> param_names_index_;
  /// blob indices for the input and the output of the net
  vector<int> net_input_blob_indices_;
  vector<int> net_output_blob_indices_;
  vector<Blob*> net_input_blobs_;
  vector<Blob*> net_output_blobs_;
  /// The parameters in the network.
  vector<shared_ptr<Blob>> params_;
  vector<shared_ptr<Blob>> learnable_params_;
  vector<shared_ptr<Blob>> learnable_params_mapped_;
  bool trained_layers_shared_;

  vector<Type> learnable_types_;
  vector<void*> learnable_params_ptrs_[2];
  GPUMemory::Workspace learnable_space_[2];
  size_t learnable_space_size_[2];
  /// layer_id->{paramss}
  std::map<size_t, std::set<int>> ltop_[2];
  size_t reduce_buckets_;

  /**
   * The mapping from params_ -> learnable_params_: we have
   * learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i]
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  vector<int> learnable_param_ids_;
  /// the learning rate multipliers for learnable_params_
  vector<float> params_lr_;
  vector<bool> has_params_lr_;
  /// the weight decay multipliers for learnable_params_
  vector<float> params_weight_decay_;
  vector<bool> has_params_decay_;
  /// The bytes of memory __planned_to_be_used__ by this net
  // TODO: clean
  size_t gpu_top_memory_data_use_, gpu_top_memory_diff_use_;
  size_t gpu_btm_memory_data_use_, gpu_btm_memory_diff_use_;
  size_t gpu_shr_memory_data_use_, gpu_shr_memory_diff_use_;
  size_t gpu_prm_memory_data_use_, gpu_prm_memory_diff_use_;
  size_t gpu_shp_memory_data_use_, gpu_shp_memory_diff_use_;
  unsigned int batch_per_solver_;
  /// Whether to compute and display debug info for the net.
  bool debug_info_;
  /// The root net that actually holds the shared layers in data parallelism
  const Net* const root_net_;
  /// Pointer to the solver being used with this net
  Solver* solver_;
  size_t solver_rank_;
  BlockingQueue<int> reduction_queue_[2];
  Flag* solver_init_flag_;
  vector<Flag*> layer_inititialized_flags_;
  NetParameter net_param_;

  size_t infer_count_;
  std::atomic_llong wgrad_sq_;
  float global_grad_scale_coeff_, global_grad_scale_param_;
  bool has_global_grad_scale_param_, global_grad_scale_adaptive_;
  /// Inner net runs on singe GPU (see recurrent layers)
  const bool inner_net_;

  static constexpr float GRAD_FACTOR = 1.E6F;

  static constexpr int END_OF_ITERATION = -1;
  static constexpr int END_OF_TRAIN = -2;

  // Range values
  vector<vector<float> > max_in_;
  vector<vector<float> > min_in_;
  vector<vector<float> > max_out_;
  vector<vector<float> > min_out_;
  vector<vector<float> > max_weights_;
  vector<vector<float> > min_weights_;
  
  DISABLE_COPY_MOVE_AND_ASSIGN(Net);
};

}  // namespace caffe

#endif  // CAFFE_NET_HPP_
